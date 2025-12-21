import re
import sys
from pathlib import Path

from transformers import MarianMTModel, MarianTokenizer

TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}")

def flush(buf, idx_map, out, tokenizer, model):
    inputs = tokenizer(buf, return_tensors="pt", padding=True, truncation=True)
    gen = model.generate(**inputs, max_new_tokens=256)
    translated = tokenizer.batch_decode(gen, skip_special_tokens=True)

    for pos, t in zip(idx_map, translated):
        newline = "\n" if out[pos].endswith("\n") else ""
        out[pos] = t + newline
    return out

def translate_vtt_lines(lines, tokenizer, model, batch_size=12):
    out = []
    buf = []
    idx_map = []

    for line in lines:
        stripped = line.strip()

        # păstrează liniile goale, header, și timpii
        if not stripped or stripped == "WEBVTT" or TIME_RE.match(stripped):
            out.append(line)
            continue

        # unele VTT-uri pot avea id-uri numerice; le păstrăm
        if stripped.isdigit():
            out.append(line)
            continue

        # linie de text -> traducem
        buf.append(stripped)
        idx_map.append(len(out))
        out.append(line)  # placeholder

        if len(buf) >= batch_size:
            out = flush(buf, idx_map, out, tokenizer, model)
            buf, idx_map = [], []

    if buf:
        out = flush(buf, idx_map, out, tokenizer, model)

    return out

def main():
    if len(sys.argv) != 3:
        print("Usage: translate_vtt_fr_ro.py input.vtt output.vtt")
        sys.exit(1)

    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])

    # Model FR -> RO (OPUS-MT)
    model_name = "Helsinki-NLP/opus-mt-fr-ro"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    lines = inp.read_text(encoding="utf-8", errors="replace").splitlines(True)
    translated = translate_vtt_lines(lines, tokenizer, model)

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("".join(translated), encoding="utf-8")

if __name__ == "__main__":
    main()
