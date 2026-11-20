# Qari Reader · قارئ

Turn scanned Arabic PDFs and page photos into clean, editable text, using **[Qari-OCR](https://huggingface.co/NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct)**. Qari-OCR is an open-source model fine-tuned for Arabic, and it is strong on tashkeel (تشكيل).

It runs on your own machine or on Google Colab's free GPU. You don't need an API key or pay per page, and your files never leave the computer the tool runs on.

- **Inputs:** PDF, JPG/PNG/WebP/TIFF (including multi-page TIFF), or a folder of page images.
- **Outputs:** `.txt`, `.docx` (right-to-left, Arabic font), `.md`, `.json`.
- **Two ways to use it:** a web app (upload, Start, download) or a command line for whole books.
- **Resumes automatically.** Each finished page is saved, so after a crash, a stop or a Colab disconnect you just run the same command again.
- **Self-checking.** If the model gets cut off or starts repeating itself, the page is re-read automatically in smaller pieces, cut through the gaps between lines. Pages that still look doubtful are listed at the end so you know where to look.
- **Page clean-up.** Margins and scanner borders are cropped. Optional extras: contrast clean-up for faded scans, and two-column detection (right column read first).

## Quickest start: Google Colab (free GPU, nothing to install)

1. Open the notebook in Colab: **[Open Qari_Reader_Colab.ipynb](https://colab.research.google.com/github/fekovic000-ops/Qari-OCR/blob/main/Qari_Reader_Colab.ipynb)**
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the cells from top to bottom. The first one installs everything from this repository.
4. Click the `gradio.live` link and use the app. It works from your phone too.

## Run on your own computer

You need Python 3.10+. An NVIDIA GPU with 8 GB or more is recommended. Apple Silicon (M1–M4) works through `mps`. A CPU alone works but takes minutes per page.

```bash
git clone https://github.com/fekovic000-ops/Qari-OCR.git
cd Qari-OCR
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[web]"
```

The first run downloads the model (about 4.5 GB) once.

**Web app**
```bash
qari-reader-web            # then open http://localhost:7860
```

**Command line**
```bash
qari-reader book.pdf                                  # whole book → output/book/book.txt + .docx
qari-reader book.pdf --pages 1-40 --format txt,md
qari-reader scans/ --enhance --columns                # folder of photos, faded, two columns
qari-reader a.pdf b.pdf c.pdf --batch 3               # several books, 3 pages at a time on the GPU
```

| Option | What it does |
|---|---|
| `--pages 1-20,25,40-` | choose pages |
| `--model v0.3` | layout-aware model with early handwriting support (v0.2, the default, is more accurate on printed text) |
| `--enhance` | clean up grey, yellowed or faded scans |
| `--columns` | split two-column pages, right column first |
| `--split 2` | always read each page in 2 bands (very dense pages) |
| `--max-pixels 2500000` | more detail for very small print |
| `--quantize 8bit` / `4bit` | less GPU memory (NVIDIA only; `pip install bitsandbytes`) |
| `--batch 2` | read several pages at once (faster on big GPUs) |
| `--no-tashkeel`, `--remove-tatweel` | strip diacritics or ـ from the output |
| `--no-markers` | leave out the `— صفحة ٣ —` page markers |
| `--redo` | ignore saved pages and read everything again |

## Which model?

| | v0.2 (default) | v0.3 |
|---|---|---|
| Printed text accuracy | best (paper: CER 0.061, WER 0.160) | lower |
| Tashkeel | best | good |
| Layout / headings | plain text | better structure |
| Handwriting | weak | early support |

## Speed (rough estimates — will vary with page density)

| Hardware | Per page |
|---|---|
| Colab T4 (free) | ~10–25 s |
| RTX 3090 / 4090 | ~4–8 s |
| Apple M-series | ~30–90 s |
| CPU only | 1–5 min |

Dense pages take longer because the model writes more text.

## Limits to know

According to the model card, Qari-OCR is weaker on handwriting, decorative calligraphy, very small (<14 pt) or very large print, heavily damaged scans, and complex multi-column layouts. `--enhance`, `--columns` and `--max-pixels` help with some of these. Always proofread pages the tool flags.

## Credits

Model: [Qari-OCR](https://arxiv.org/abs/2506.02295) by NAMAA-Space (Wasfy et al., 2025), built on Qwen2-VL-2B. Check the model's licence before commercial use.

## Tests

```bash
pip install -e ".[dev]" && pytest      # offline tests, no GPU or model needed
```
