"""Qari Reader — Arabic OCR for scanned PDFs and images, built on Qari-OCR."""

__version__ = "0.1.0"

MODELS = {
    # Best raw-text accuracy, strongest on tashkeel (paper: CER 0.061 for v0.2).
    "v0.2": "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct",
    # Better at layout / markdown structure and early handwriting support, lower raw accuracy.
    "v0.3": "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
}
DEFAULT_MODEL = "v0.2"
