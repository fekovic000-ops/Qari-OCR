"""The Qari-OCR model wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PIL import Image

from . import MODELS, DEFAULT_MODEL

log = logging.getLogger(__name__)

# The prompt the Qari-OCR authors trained and published with. Keep it as is:
# changing it tends to lower accuracy on a small fine-tuned model.
QARI_PROMPT = (
    "Below is the image of one page of a document, as well as some raw textual content "
    "that was previously extracted for it. Just return the plain text representation of "
    "this document as if you were reading it naturally. Do not hallucinate."
)


@dataclass
class EngineConfig:
    model: str = DEFAULT_MODEL           # "v0.2", "v0.3" or a Hugging Face repo id
    device: str = "auto"                 # "auto", "cuda", "mps" or "cpu"
    quantize: str | None = None          # None, "8bit" or "4bit" (needs CUDA + bitsandbytes)
    max_new_tokens: int = 2000           # model card default
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1600 * 28 * 28     # ~1.25 MP; raise for tiny print, lower if you run out of memory
    batch_size: int = 1
    repetition_penalty: float = 1.0      # >1.0 discourages loops but can alter repeated Arabic words
    extra: dict = field(default_factory=dict)


@dataclass
class OcrResult:
    text: str
    truncated: bool   # hit max_new_tokens — the page may be cut off


class QariEngine:
    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self._load()

    # ------------------------------------------------------------ loading ---
    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        repo = MODELS.get(self.cfg.model, self.cfg.model)
        self.repo = repo

        device = self.cfg.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        if device == "cpu":
            log.warning("No GPU found — running on CPU. Expect roughly 1–5 minutes per page.")

        kwargs: dict = {}
        if self.cfg.quantize and device == "cuda":
            from transformers import BitsAndBytesConfig
            if self.cfg.quantize == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            elif self.cfg.quantize == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        elif device == "cuda":
            major = torch.cuda.get_device_capability()[0]
            kwargs["torch_dtype"] = torch.bfloat16 if major >= 8 else torch.float16
            kwargs["device_map"] = "auto"
        elif device == "mps":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        log.info("Loading %s on %s …", repo, device)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(repo, **kwargs)
        if device in ("mps", "cpu") and "device_map" not in kwargs:
            self.model.to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            repo, min_pixels=self.cfg.min_pixels, max_pixels=self.cfg.max_pixels)
        # left padding is required for batched generation with decoder-only models
        self.processor.tokenizer.padding_side = "left"
        self._eos = self.processor.tokenizer.eos_token_id

    # ---------------------------------------------------------- inference ---
    def read(self, images: list[Image.Image]) -> list[OcrResult]:
        """OCR a list of images, in batches of cfg.batch_size."""
        results: list[OcrResult] = []
        bs = max(1, self.cfg.batch_size)
        for i in range(0, len(images), bs):
            results.extend(self._read_batch(images[i:i + bs]))
        return results

    def _read_batch(self, images: list[Image.Image]) -> list[OcrResult]:
        import torch
        from qwen_vl_utils import process_vision_info

        msgs = [[{"role": "user", "content": [
                    {"type": "image", "image": img,
                     "min_pixels": self.cfg.min_pixels, "max_pixels": self.cfg.max_pixels},
                    {"type": "text", "text": QARI_PROMPT}]}]
                for img in images]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,              # deterministic: OCR should not be creative
                repetition_penalty=self.cfg.repetition_penalty,
                **self.cfg.extra,
            )

        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        results = []
        for ids, txt in zip(trimmed, decoded):
            ids = ids.tolist()
            # strip trailing padding before checking length
            while ids and ids[-1] == self.processor.tokenizer.pad_token_id:
                ids.pop()
            truncated = len(ids) >= self.cfg.max_new_tokens and (not ids or ids[-1] != self._eos)
            results.append(OcrResult(txt, truncated))
        return results
