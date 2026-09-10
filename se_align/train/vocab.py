"""Combined text+audio vocabulary for the Phase-2 Qwen + CV3 model.

Adapted from EchoMind's ``VocabConfig`` (settings.py) but specialised to
CosyVoice3 (single code layer, 6561 FSQ tokens) on a Qwen2.5 text backbone.

Layout of the single resized embedding / LM head (size = ``total_vocabsize``):

    [0 .. padded_text_vocabsize)              -> text tokens   (Qwen2.5 + text specials)
    [padded_text_vocabsize .. total_vocabsize) -> audio tokens  (CV3 + audio specials)

`layershift(a)` maps a raw audio id ``a`` into the combined space (``a + audio_shift``)
for **input_ids / embedding lookup**. Targets in the loss stay **raw** (0-based in
their own slice). code_layer is 1 for CV3, so there is exactly one audio stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SEVocabConfig:
    # Qwen2.5 embedding rows (Qwen2.5-*-Instruct has 151936 embedding rows).
    text_vocabsize: int = 151936
    text_specialtokens: int = 64
    # CosyVoice3 FSQ codebook.
    audio_vocabsize: int = 6561
    audio_specialtokens: int = 64
    code_layer: int = 1

    padded_text_vocabsize: int = field(init=False)
    padded_audio_vocabsize: int = field(init=False)
    total_audio_vocabsize: int = field(init=False)
    total_vocabsize: int = field(init=False)
    audio_shift: int = field(init=False)

    # text special tokens (offsets into the text special region)
    eot: int = field(init=False)
    pad_t: int = field(init=False)
    input_t: int = field(init=False)
    answer_t: int = field(init=False)
    asr: int = field(init=False)

    # audio special tokens (raw, within the audio slice)
    eoa: int = field(init=False)
    pad_a: int = field(init=False)
    input_a: int = field(init=False)
    answer_a: int = field(init=False)
    split: int = field(init=False)

    def __post_init__(self) -> None:
        self.padded_text_vocabsize = self.text_vocabsize + self.text_specialtokens
        self.padded_audio_vocabsize = self.audio_vocabsize + self.audio_specialtokens
        self.total_audio_vocabsize = self.padded_audio_vocabsize * self.code_layer
        self.total_vocabsize = self.padded_text_vocabsize + self.total_audio_vocabsize
        self.audio_shift = self.padded_text_vocabsize

        self.eot = self.text_vocabsize
        self.pad_t = self.text_vocabsize + 1
        self.input_t = self.text_vocabsize + 2
        self.answer_t = self.text_vocabsize + 3
        self.asr = self.text_vocabsize + 4

        self.eoa = self.audio_vocabsize
        self.pad_a = self.audio_vocabsize + 1
        self.input_a = self.audio_vocabsize + 2
        self.answer_a = self.audio_vocabsize + 3
        self.split = self.audio_vocabsize + 4

    # layer is always 0 for CV3; kept for parity with multi-codebook codecs.
    def layershift(self, audio_id: int, layer: int = 0) -> int:
        """Raw audio id -> combined-vocab id for embedding lookup."""
        return audio_id + self.audio_shift + layer * self.padded_audio_vocabsize
