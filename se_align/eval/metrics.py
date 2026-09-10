"""Speech-enhancement metrics for the reconstruction baseline.

Two families:
  * intrusive (need a clean reference): PESQ-wb, STOI, ESTOI, SI-SDR
  * non-intrusive (reference-free):     DNSMOS (SIG/BAK/OVL/P808), UTMOS
  * task:                               WER (whisper ASR), SECS (spk cosine)

Every metric degrades gracefully: if its backend is missing it returns ``None``
and logs once, so a partial environment still produces a table.

Reminder (recorded in the README): CosyVoice3 emits 25 Hz *semantic* tokens and
regenerates audio, so PESQ/SI-SDR are expected to be low even at the ceiling.
DNSMOS/UTMOS + WER + SECS are the primary judgement axes.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..utils.common import get_logger

LOG = get_logger("metrics")
EPS = 1e-8


# ----------------------------- intrusive ------------------------------- #
def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + EPS))
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + EPS) / (np.dot(noise, noise) + EPS)))


def pesq_wb(ref: np.ndarray, est: np.ndarray, sr: int = 16000) -> Optional[float]:
    try:
        from pesq import pesq

        return float(pesq(sr, ref, est, "wb"))
    except Exception as e:  # noqa: BLE001
        LOG.debug("pesq failed: %s", e)
        return None


def stoi_score(ref: np.ndarray, est: np.ndarray, sr: int = 16000, extended: bool = False) -> Optional[float]:
    try:
        from pystoi import stoi

        return float(stoi(ref, est, sr, extended=extended))
    except Exception as e:  # noqa: BLE001
        LOG.debug("stoi failed: %s", e)
        return None


# --------------------------- non-intrusive ----------------------------- #
def _preload_nvidia_libs() -> None:
    """dlopen(RTLD_GLOBAL) the pip-installed CUDA libs onnxruntime-gpu needs.

    The CUDAExecutionProvider dlopens libonnxruntime_providers_cuda.so, which
    links libcublas.so.12 / libcudnn.so.9 / libcufft.so.11 / libcudart.so.12.
    Those live under site-packages/nvidia/*/lib (pip wheels), not on the system
    LD_LIBRARY_PATH -- preloading them here makes GPU work in any process.
    """
    import ctypes
    import glob as _glob
    import os
    import sysconfig

    sp = sysconfig.get_paths()["purelib"]
    for pat in ("cublas/lib/libcublas.so.*", "cublas/lib/libcublasLt.so.*",
                "cudnn/lib/libcudnn.so.*", "cufft/lib/libcufft.so.*",
                "cuda_runtime/lib/libcudart.so.*"):
        for so_path in sorted(_glob.glob(os.path.join(sp, "nvidia", pat))):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class _DNSMOS:
    def __init__(self, use_gpu: bool = False) -> None:
        import os

        from speechmos import dnsmos

        self._run = dnsmos.run
        # DNSMOS onnxruntime defaults to one intra-op thread PER CORE -> ~100 threads
        # per session. Under many parallel eval shards this explodes (1000+ threads)
        # and thrashes the CPU. Monkeypatch InferenceSession while the speechmos
        # singleton is first constructed so BOTH sessions (sig_bak_ovr + p808
        # model_v8 -- note: DNSMOS keeps no p808_model_path attribute) are built
        # ONCE with a small thread cap (env DNSMOS_THREADS, default 2) and, when
        # use_gpu, on CUDAExecutionProvider (needs nvidia cublas/cudnn/cufft/
        # cuda_runtime pip libs on LD_LIBRARY_PATH).
        nthr = int(os.environ.get("DNSMOS_THREADS", "2"))
        if use_gpu:
            _preload_nvidia_libs()  # so CUDAExecutionProvider finds cublas etc.
        try:
            import onnxruntime as ort
            import speechmos.dnsmos as dm
            so = ort.SessionOptions()
            so.intra_op_num_threads = nthr
            so.inter_op_num_threads = 1
            prov = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
                    else ["CPUExecutionProvider"])
            if getattr(dm, "dnsmos", None) is None:
                orig = ort.InferenceSession
                ort.InferenceSession = (  # capped/GPU sessions from the start
                    lambda path, *a, **kw: orig(path, sess_options=so, providers=prov))
                try:
                    dm.run(np.zeros(16000, dtype=np.float32), sr=16000)
                finally:
                    ort.InferenceSession = orig
            else:  # singleton already exists (built elsewhere): rebuild both
                import os as _os
                mdir = _os.path.dirname(dm.dnsmos.primary_model_path)
                dm.dnsmos.onnx_sess = ort.InferenceSession(
                    dm.dnsmos.primary_model_path, sess_options=so, providers=prov)
                dm.dnsmos.p808_onnx_sess = ort.InferenceSession(
                    _os.path.join(mdir, "model_v8.onnx"), sess_options=so, providers=prov)
            self._providers = dm.dnsmos.onnx_sess.get_providers()
        except Exception:  # noqa: BLE001 — fall back to defaults silently
            self._providers = ["?"]

    def __call__(self, wav16: np.ndarray) -> Dict[str, float]:
        # vocoder output can overshoot [-1,1]; DNSMOS hard-requires that range.
        wav16 = np.clip(wav16.astype(np.float32), -1.0, 1.0)
        r = self._run(wav16, sr=16000)
        return {
            "dnsmos_sig": float(r["sig_mos"]),
            "dnsmos_bak": float(r["bak_mos"]),
            "dnsmos_ovl": float(r["ovrl_mos"]),
            "dnsmos_p808": float(r["p808_mos"]),
        }


class _UTMOS:
    """torch.hub SpeechMOS UTMOS22 (opt-in: downloads external code).

    The hub repo ships an internal package also named ``speechmos`` which clashes
    with the pip ``speechmos`` (DNSMOS). We isolate the load: stash the installed
    ``speechmos.*`` modules, prepend the hub repo dir, load, then restore — so both
    UTMOS and DNSMOS work in the same process regardless of import order.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        import os
        import sys

        import torch

        self._torch = torch
        self._device = device or "cpu"
        repo = os.path.join(torch.hub.get_dir(), "tarepan_SpeechMOS_v1.2.0")
        stashed = {k: sys.modules.pop(k) for k in list(sys.modules)
                   if k == "speechmos" or k.startswith("speechmos.")}
        sys.path.insert(0, repo)
        try:
            self._model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        finally:
            if repo in sys.path:
                sys.path.remove(repo)
            for k in [m for m in list(sys.modules) if m == "speechmos" or m.startswith("speechmos.")]:
                del sys.modules[k]
            sys.modules.update(stashed)  # restore the pip speechmos (DNSMOS)
        if self._device != "cpu":
            self._model = self._model.to(self._device)

    def __call__(self, wav16: np.ndarray) -> float:
        x = self._torch.from_numpy(wav16.astype(np.float32)).unsqueeze(0).to(self._device)
        return float(self._model(x, 16000))


class _ASRWer:
    """Whisper-family WER via transformers (no extra ASR dependency)."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        import torch
        from transformers import pipeline

        dev = 0 if (device != "cpu" and torch.cuda.is_available()) else -1
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=dev,
            torch_dtype=torch.float16 if dev >= 0 else torch.float32,
        )

    def transcribe(self, wav16: np.ndarray) -> str:
        out = self._pipe({"raw": wav16.astype(np.float32), "sampling_rate": 16000})
        return (out.get("text") or "").strip()

    def wer(self, wav16: np.ndarray, ref_text: str) -> Optional[float]:
        try:
            import jiwer

            hyp = self.transcribe(wav16)
            tr = jiwer.Compose(
                [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
            )
            return float(jiwer.wer(tr(ref_text), tr(hyp)))
        except Exception as e:  # noqa: BLE001
            LOG.debug("wer failed: %s", e)
            return None


def secs(ref_emb: np.ndarray, est_emb: np.ndarray) -> float:
    """Speaker-embedding cosine similarity (CAMPPlus 192-d)."""
    a = np.asarray(ref_emb, dtype=np.float64).reshape(-1)
    b = np.asarray(est_emb, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


# ----------------- SSL-feature / phoneme / VQ metrics ------------------ #
class _SpeechBERTScore:
    """BERTScore over SSL features (Saeki et al. 2024).

    Greedy cosine matching between reference and generated frame features from a
    self-supervised model (default the cached wav2vec2-base; wavlm-large layer 14
    is the paper's recommendation). Reports the F1. Reference-aware.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base", layer: int = 8, device: Optional[str] = None):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = layer
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device).eval()

    def _feats(self, wav16: np.ndarray):
        import torch

        x = torch.from_numpy(wav16.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hs = self.model(x).hidden_states[self.layer][0]  # [T, D]
        return torch.nn.functional.normalize(hs, dim=-1)

    def __call__(self, ref16: np.ndarray, est16: np.ndarray) -> float:
        r, g = self._feats(ref16), self._feats(est16)
        if r.shape[0] == 0 or g.shape[0] == 0:
            return 0.0
        sim = g @ r.T  # [Tg, Tr] cosine
        precision = sim.max(dim=1).values.mean()
        recall = sim.max(dim=0).values.mean()
        f1 = 2 * precision * recall / (precision + recall + EPS)
        return float(f1)


class _PhonemeRecognizer:
    """wav2vec2 phoneme CTC -> phoneme string (for Levenshtein phoneme similarity).

    The default espeak model needs the ``phonemizer`` lib + the ``espeak-ng``
    system binary (``apt install espeak-ng``). Without them this stays disabled.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft", device: Optional[str] = None):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2PhonemeCTCTokenizer, AutoModelForCTC

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name).to(self.device).eval()

    def phonemes(self, wav16: np.ndarray) -> list:
        import torch

        iv = self.fe(wav16.astype(np.float32), sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            ids = self.model(iv).logits.argmax(-1)
        txt = self.tok.batch_decode(ids)[0]
        return txt.split()


def phoneme_levenshtein_sim(ref_ph: list, est_ph: list) -> float:
    """1 - normalised phoneme edit distance (1.0 = identical phoneme sequence)."""
    if not ref_ph and not est_ph:
        return 1.0
    try:
        import editdistance

        d = int(editdistance.eval(ref_ph, est_ph))
    except ImportError:
        # Exact two-row Wagner-Fischer fallback. This keeps the registered LPS
        # definition available in minimal evaluation environments without
        # changing its edit-distance semantics.
        previous = list(range(len(est_ph) + 1))
        for i, ref_value in enumerate(ref_ph, 1):
            current = [i]
            for j, est_value in enumerate(est_ph, 1):
                current.append(min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_value != est_value),
                ))
            previous = current
        d = previous[-1]
    return float(1.0 - d / max(len(ref_ph), len(est_ph), 1))


def _stft_magnitude(x, hop_size: int, fft_size: int = 512, win_length: int = 512):
    import torch

    win = torch.hann_window(win_length).to(x.device)
    st = torch.stft(x, fft_size, hop_size, win_length, window=win, return_complex=False)
    real, imag = st[..., 0], st[..., 1]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class _VQScore:
    """Reference-free VQScore (Fu et al. 2024): cosine(z, zq) of a VQ-VAE quality
    estimator. Loads the official repo's ``VQVAE_QE`` + checkpoint (JasonSWFu/VQscore).

    Needs cfg: ``vqscore_repo`` (repo dir), ``vqscore_config`` (QE yaml),
    ``vqscore_ckpt`` (the QE .pkl). Higher = better speech quality.
    """

    def __init__(self, repo: str, config_path: str, ckpt: str, device: Optional[str] = None):
        import sys

        import torch
        import yaml

        if repo not in sys.path:
            sys.path.insert(0, repo)
        from models.VQVAE_models import VQVAE_QE

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.input_transform = cfg.get("input_transform", "None")
        self.hop = 256
        self.model = VQVAE_QE(**cfg["VQVAE_params"]).to(self.device).eval()
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device)["model"]["VQVAE"])

    def __call__(self, wav16: np.ndarray) -> float:
        import torch

        x = torch.from_numpy(wav16.astype(np.float32)).unsqueeze(0).to(self.device)
        sp = _stft_magnitude(x, hop_size=self.hop)
        if self.input_transform == "log1p":
            sp = torch.log1p(sp)
        with torch.no_grad():
            z = self.model.CNN_1D_encoder(sp)
            zq, *_ = self.model.quantizer(z, stochastic=False, update=False)
            eps = 1e-5
            zt = z.transpose(2, 1)
            cos = torch.sum(
                zt / (zt.norm(p=2, dim=-1, keepdim=True) + eps)
                * zq / (zq.norm(p=2, dim=-1, keepdim=True) + eps), dim=-1)
            return float(cos.mean())


# --------------------------- bundle / driver --------------------------- #
class MetricBundle:
    """Lazily-loaded collection of metrics, gated by a cfg.metrics toggle map."""

    def __init__(self, toggles: dict, asr_model: Optional[str] = None, device: Optional[str] = None,
                 gpu_extra: bool = False) -> None:
        self.t = dict(toggles or {})
        self.asr_model = asr_model
        self.device = device
        # opt-in: run UTMOS + DNSMOS on GPU too (default OFF -> identical to prior CPU
        # behaviour, so a running campaign is unaffected). Future evals pass gpu_extra=True.
        self.gpu_extra = gpu_extra and (device not in (None, "cpu"))
        self._dnsmos = None
        self._utmos = None
        self._asr = None
        self._utmos_failed = False
        self._sbert = None
        self._phone = None
        self._vq = None

    def _get_dnsmos(self):
        if self._dnsmos is None:
            try:
                self._dnsmos = _DNSMOS(use_gpu=self.gpu_extra)
            except Exception as e:  # noqa: BLE001
                LOG.warning("DNSMOS unavailable: %s", e)
                self.t["dnsmos"] = False
        return self._dnsmos

    def _get_utmos(self):
        if self._utmos is None and not self._utmos_failed:
            try:
                self._utmos = _UTMOS(device=self.device if self.gpu_extra else None)
            except Exception as e:  # noqa: BLE001
                LOG.warning("UTMOS unavailable (needs torch.hub opt-in): %s", e)
                self._utmos_failed = True
                self.t["utmos"] = False
        return self._utmos

    def _get_asr(self):
        if self._asr is None and self.asr_model:
            try:
                self._asr = _ASRWer(self.asr_model, self.device)
            except Exception as e:  # noqa: BLE001
                LOG.warning("ASR/WER unavailable: %s", e)
                self.t["wer"] = False
        return self._asr

    def _get_sbert(self):
        if self._sbert is None:
            try:
                self._sbert = _SpeechBERTScore(
                    self.t.get("speechbertscore_model", "facebook/wav2vec2-base"),
                    layer=self.t.get("speechbertscore_layer", 8), device=self.device)
            except Exception as e:  # noqa: BLE001
                LOG.warning("SpeechBERTScore unavailable: %s", e)
                self.t["speechbertscore"] = False
        return self._sbert

    def _get_phone(self):
        if self._phone is None:
            try:
                self._phone = _PhonemeRecognizer(
                    self.t.get("phoneme_model", "facebook/wav2vec2-lv-60-espeak-cv-ft"),
                    device=self.device)
            except Exception as e:  # noqa: BLE001
                LOG.warning("phoneme recognizer unavailable: %s", e)
                self.t["phoneme_sim"] = False
        return self._phone

    def _get_vq(self):
        if self._vq is None:
            try:
                repo = self.t.get("vqscore_repo")
                cfgp = self.t.get("vqscore_config")
                ckpt = self.t.get("vqscore_ckpt")
                if not (repo and cfgp and ckpt):
                    raise RuntimeError("set metrics.vqscore_repo/_config/_ckpt")
                self._vq = _VQScore(repo, cfgp, ckpt, device=self.device)
            except Exception as e:  # noqa: BLE001
                LOG.warning("VQScore unavailable: %s", e)
                self.t["vqscore"] = False
        return self._vq

    def score(
        self,
        ref16: Optional[np.ndarray],
        est16: np.ndarray,
        ref_text: Optional[str] = None,
        ref_emb: Optional[np.ndarray] = None,
        est_emb: Optional[np.ndarray] = None,
    ) -> Dict[str, Optional[float]]:
        """Score one utterance; ``ref16`` may be None for non-intrusive only."""
        out: Dict[str, Optional[float]] = {}
        if ref16 is not None:
            n = min(len(ref16), len(est16))
            r, e = ref16[:n], est16[:n]
            if self.t.get("pesq"):
                out["pesq"] = pesq_wb(r, e)
            if self.t.get("stoi"):
                out["stoi"] = stoi_score(r, e, extended=False)
            if self.t.get("estoi"):
                out["estoi"] = stoi_score(r, e, extended=True)
            if self.t.get("si_sdr"):
                out["si_sdr"] = si_sdr(r, e)
        if self.t.get("dnsmos"):
            d = self._get_dnsmos()
            if d is not None:
                try:
                    out.update(d(est16))
                except Exception as e:  # noqa: BLE001
                    LOG.debug("dnsmos failed: %s", e)
        if self.t.get("utmos"):
            u = self._get_utmos()
            if u is not None:
                out["utmos"] = u(est16)
        if self.t.get("wer") and ref_text is not None:
            a = self._get_asr()
            if a is not None:
                out["wer"] = a.wer(est16, ref_text)
        if self.t.get("secs") and ref_emb is not None and est_emb is not None:
            out["secs"] = secs(ref_emb, est_emb)
        # SSL-feature / phoneme / VQ
        if self.t.get("speechbertscore") and ref16 is not None:
            s = self._get_sbert()
            if s is not None:
                try:
                    out["speechbertscore"] = s(ref16, est16)
                except Exception as e:  # noqa: BLE001
                    LOG.debug("speechbertscore failed: %s", e)
        if self.t.get("phoneme_sim") and ref16 is not None:
            p = self._get_phone()
            if p is not None:
                try:
                    out["phoneme_sim"] = phoneme_levenshtein_sim(p.phonemes(ref16), p.phonemes(est16))
                except Exception as e:  # noqa: BLE001
                    LOG.debug("phoneme_sim failed: %s", e)
        if self.t.get("vqscore"):
            vq = self._get_vq()
            if vq is not None:
                try:
                    out["vqscore"] = vq(est16)
                except Exception as e:  # noqa: BLE001
                    LOG.debug("vqscore failed: %s", e)
        return out


def aggregate(rows) -> Dict[str, float]:
    """Mean over per-utt metric dicts, ignoring None/missing.

    Rows are deduplicated by utt_id (last wins) so that shard files written
    under different --num-shards partitions don't double-count utterances.
    """
    rows = list(rows)
    if rows and isinstance(rows[0], dict) and "utt_id" in rows[0]:
        rows = list({r.get("utt_id", id(r)): r for r in rows}.values())
    keys = set()
    for r in rows:
        keys.update(r.keys())
    agg = {}
    for k in keys:
        vals = [
            r[k] for r in rows
            if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
        ]
        if vals:
            agg[k] = float(np.mean(vals))
    return agg
