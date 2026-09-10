"""Official CosyVoice3 S3 tokenizer and lazy Flow/HiFT adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

from ..utils.audio import resample
from ..utils.common import set_seed

TokenInput = Union[Sequence[int], torch.Tensor]


def _ensure_cv3_on_path(cv3_root: str) -> None:
    root = str(Path(cv3_root).resolve())
    matcha = str(Path(root) / "third_party" / "Matcha-TTS")
    for path in (root, matcha):
        if path not in sys.path:
            sys.path.insert(0, path)


class CosyVoice3S3Tokenizer:
    """Encode 16 kHz audio with the official CosyVoice3 ONNX S3 model."""

    sample_rate = 16000
    token_rate_hz = 25
    vocab_size = 6561

    def __init__(self, model_dir: str, provider: str = "CPUExecutionProvider") -> None:
        import onnxruntime as ort

        model_path = Path(model_dir) / "speech_tokenizer_v3.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        providers = (
            [provider, "CPUExecutionProvider"]
            if provider != "CPUExecutionProvider" else [provider]
        )
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )
        active_providers = self.session.get_providers()
        if provider not in active_providers:
            raise RuntimeError(
                f"requested ONNX provider {provider}, active providers: {active_providers}"
            )
        self.provider = provider
        self.feat_name, self.length_name = [item.name for item in self.session.get_inputs()]

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        import whisper

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = resample(wav.float(), int(sr), self.sample_rate).contiguous()
        duration = wav.shape[-1] / self.sample_rate
        if duration > 30:
            raise ValueError(f"CosyVoice3 S3 supports at most 30 seconds, got {duration:.2f}")
        feat = whisper.log_mel_spectrogram(wav, n_mels=128)
        tokens = self.session.run(
            None,
            {
                self.feat_name: feat.cpu().numpy(),
                self.length_name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0].reshape(-1).astype(np.int64, copy=False)
        if tokens.size == 0:
            raise RuntimeError("CosyVoice3 returned no S3 tokens")
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise RuntimeError(
                f"CosyVoice3 returned invalid token range [{tokens.min()}, {tokens.max()}]"
            )
        return torch.from_numpy(tokens.copy())


class CosyVoice3Codec:
    """Stable encode/speaker/decode API backed only by official CosyVoice code."""

    def __init__(self, model_dir: str, cv3_root: str,
                 repo_dir: Optional[str] = None, device: Optional[str] = None,
                 flow_steps: int = 10, seed: int = 1986) -> None:
        del repo_dir
        _ensure_cv3_on_path(cv3_root)
        self.model_dir = str(Path(model_dir).resolve())
        self.cv3_root = str(Path(cv3_root).resolve())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.flow_steps = int(flow_steps)
        self.seed = int(seed)
        self.sample_rate = 24000
        self.encode_sr = 16000
        self.token_mel_ratio = 2
        self.spk_emb_dim = 192
        self.tokenizer = CosyVoice3S3Tokenizer(self.model_dir)
        self._official = None

        import onnxruntime as ort

        campplus = Path(self.model_dir) / "campplus.onnx"
        self._campplus = ort.InferenceSession(
            str(campplus), providers=["CPUExecutionProvider"]
        )
        self._campplus_input = self._campplus.get_inputs()[0].name

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        return self.tokenizer.encode(wav, sr)

    @torch.no_grad()
    def extract_spk_emb(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        wav16 = self._to_mono16k(wav, sr)
        feat = kaldi.fbank(
            wav16,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self._campplus.run(
            None, {self._campplus_input: feat.unsqueeze(0).cpu().numpy()}
        )[0].reshape(-1)
        return torch.from_numpy(embedding.astype(np.float32, copy=False).copy())

    def _load_official(self):
        if self._official is not None:
            return self._official
        from cosyvoice.cli.cosyvoice import AutoModel

        original_cuda_available = torch.cuda.is_available
        if self.device.type == "cpu":
            torch.cuda.is_available = lambda: False
        try:
            self._official = AutoModel(
                model_dir=self.model_dir,
                load_trt=False,
                load_vllm=False,
                fp16=self.device.type == "cuda",
            )
        finally:
            torch.cuda.is_available = original_cuda_available
        return self._official

    @torch.no_grad()
    def prepare_prompt(self, prompt_wav: Optional[torch.Tensor],
                       prompt_sr: Optional[int],
                       spk_emb: Optional[torch.Tensor] = None,
                       prompt_strategy: str = "self"):
        return self._build_prompt(prompt_wav, prompt_sr, spk_emb, prompt_strategy)

    @torch.no_grad()
    def decode(self, tokens: TokenInput, prompt_wav: Optional[torch.Tensor] = None,
               prompt_sr: Optional[int] = None,
               spk_emb: Optional[torch.Tensor] = None,
               prompt_strategy: str = "self", steps: Optional[int] = None,
               seed: Optional[int] = None, prepared_prompt=None) -> Tuple[torch.Tensor, int]:
        if steps not in (None, self.flow_steps):
            raise ValueError(f"CosyVoice3 uses a fixed {self.flow_steps}-step flow schedule")
        set_seed(self.seed if seed is None else int(seed))
        official = self._load_official()
        prompt_token, prompt_feat, embedding = (
            prepared_prompt if prepared_prompt is not None else
            self._build_prompt(prompt_wav, prompt_sr, spk_emb, prompt_strategy)
        )
        token = self._as_token_tensor(tokens).to(self.device)
        prompt_token = prompt_token.to(self.device)
        prompt_feat = prompt_feat.to(self.device)
        embedding = embedding.to(self.device)
        flow = official.model.flow
        hift = official.model.hift
        mel, _ = flow.inference(
            token=token,
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32, device=self.device),
            prompt_token=prompt_token,
            prompt_token_len=torch.tensor(
                [prompt_token.shape[1]], dtype=torch.int32, device=self.device
            ),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor(
                [prompt_feat.shape[1]], dtype=torch.int32, device=self.device
            ),
            embedding=embedding,
            streaming=False,
            finalize=True,
        )
        speech, _ = hift.inference(speech_feat=mel, finalize=True)
        return speech.detach().cpu().reshape(1, -1).float(), self.sample_rate

    def _to_mono16k(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return resample(wav.float(), int(sr), self.encode_sr).contiguous()

    @staticmethod
    def _as_token_tensor(tokens: TokenInput) -> torch.Tensor:
        values = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(list(tokens))
        return values.reshape(1, -1).to(torch.int32)

    def _build_prompt(self, prompt_wav: Optional[torch.Tensor],
                      prompt_sr: Optional[int], spk_emb: Optional[torch.Tensor],
                      strategy: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if strategy == "zero":
            if spk_emb is None and prompt_wav is None:
                raise ValueError("zero prompt needs spk_emb or prompt_wav")
            embedding = (
                spk_emb.reshape(1, -1).float() if spk_emb is not None else
                self.extract_spk_emb(prompt_wav, int(prompt_sr)).reshape(1, -1)
            )
            return (
                torch.zeros(1, 0, dtype=torch.int32),
                torch.zeros(1, 0, 80, dtype=torch.float32),
                embedding,
            )
        if strategy != "self" or prompt_wav is None or prompt_sr is None:
            raise ValueError("self prompt needs prompt_wav and prompt_sr")
        official = self._load_official()
        wav16 = self._to_mono16k(prompt_wav, prompt_sr)
        wav24 = resample(prompt_wav.reshape(1, -1).float(), int(prompt_sr), self.sample_rate)
        prompt_token = self.encode(wav16, 16000).reshape(1, -1).to(torch.int32)
        feat = official.frontend.feat_extractor(wav24).squeeze(0).transpose(0, 1)
        token_len = min(feat.shape[0] // self.token_mel_ratio, prompt_token.shape[1])
        prompt_token = prompt_token[:, :token_len]
        prompt_feat = feat[:self.token_mel_ratio * token_len].unsqueeze(0)
        embedding = (
            spk_emb.reshape(1, -1).float() if spk_emb is not None else
            self.extract_spk_emb(wav16, 16000).reshape(1, -1)
        )
        return prompt_token, prompt_feat, embedding


def build_codec_from_config(cfg) -> CosyVoice3Codec:
    cv3 = cfg["cosyvoice3"]
    return CosyVoice3Codec(
        model_dir=cv3["model_dir"],
        cv3_root=cv3["cv3_root"],
        repo_dir=cv3.get("repo_dir"),
        device=cv3.get("device"),
        flow_steps=cv3.get("flow_steps", 10),
        seed=cfg.get("seed", 1986),
    )
