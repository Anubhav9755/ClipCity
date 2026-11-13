#!/usr/bin/env python3
"""
clipper.py - turn a long video / stream recording into captioned short clips.

Pipeline
  1. Extract audio and measure loudness for every second
  2. Transcribe with faster-whisper (word-level timestamps)
  3. Score every candidate window (loud peaks + excited words + speech pace)
  4. Pick the best non-overlapping clips
  5. For each clip: cut, reframe (any size), burn animated captions,
     normalise loudness, export with one high-quality encode

Usage examples
  python clipper.py stream.mp4
  python clipper.py stream.mp4 --num-clips 8 --format vertical --model small
  python clipper.py stream.mp4 --size 1080x1350 --layout crop --crop-x 0.7
  python clipper.py stream.mp4 --gpu --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Settings you can tweak
# ----------------------------------------------------------------------------
SIZES = {
    "vertical": (1080, 1920),   # Shorts / Reels / TikTok (9:16)
    "portrait": (1080, 1350),   # Instagram feed (4:5)
    "square": (1080, 1080),     # 1:1
    "landscape": (1920, 1080),  # YouTube (16:9)
}

# Words that often show up in exciting moments (English + Hinglish).
# Add your own - this list is the easiest way to improve highlight detection.
EXCITE_RE = re.compile(
    r"\b(oh my god|omg|no way|let'?s go|insane|crazy|clutch|holy|unbelievable|"
    r"what the|bro|yo+|dead|wasted|busted|wanted|cops?|police|crash(ed)?|"
    r"explod\w*|help|run|kill\w*|win|won|lost|"
    r"arre|yaar|bhai|kya baat|pagal|zabardast|kamaal|mast)\b",
    re.IGNORECASE,
)

HIGHLIGHT_COLOR = "&H0000FFFF&"  # yellow (ASS uses BGR)
BASE_COLOR = "&H00FFFFFF&"       # white


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def run(cmd, cwd=None):
    """Run a command; stop with a readable error if it fails."""
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-12:])
        sys.exit(f"\nCommand failed: {' '.join(map(str, cmd[:3]))} ...\n{tail}")
    return p


def check_tools():
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            sys.exit(f"'{tool}' not found. Install FFmpeg and make sure it is on your PATH "
                     f"(Windows: winget install Gyan.FFmpeg).")


def probe_duration(path: Path) -> float:
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    return float(p.stdout.strip())


def safe_name(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")[:40] or "video"


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def fmt_clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{(t % 3600) // 60:02d}:{t % 60:02d}"


# ----------------------------------------------------------------------------
# Step 1: audio + loudness
# ----------------------------------------------------------------------------
def extract_audio(video: Path, wav: Path) -> bool:
    p = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        capture_output=True, text=True)
    return p.returncode == 0 and wav.exists()


def loudness_z(wav: Path | None, duration: float) -> np.ndarray:
    """Loudness (dB) for every second, as z-scores. Reads 1 second at a time
    so a 3-hour stream does not need much RAM."""
    n = int(math.ceil(duration)) + 2
    if wav is None or not wav.exists():
        return np.zeros(n, dtype=np.float32)
    db = []
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        while True:
            raw = w.readframes(sr)
            if not raw:
                break
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            db.append(20.0 * math.log10(rms + 1e-6))
    arr = np.array(db, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(n, dtype=np.float32)
    z = (arr - arr.mean()) / (arr.std() + 1e-6)
    if z.size < n:
        z = np.pad(z, (0, n - z.size), constant_values=float(z.min()))
    return z


# ----------------------------------------------------------------------------
# Step 2: transcription
# ----------------------------------------------------------------------------
def transcribe(wav: Path, duration: float, model_size: str, language: str | None, device: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed. Run: pip install faster-whisper")

    compute = "int8" if device == "cpu" else "float16"
    print(f"[2/5] Transcribing with Whisper '{model_size}' on {device} "
          f"(first run downloads the model)...")
    model = WhisperModel(model_size, device=device, compute_type=compute)
    seg_iter, info = model.transcribe(
        str(wav), language=language, word_timestamps=True, vad_filter=True, beam_size=1)
    print(f"      language: {info.language}")

    segments = []
    for s in seg_iter:
        words = [{"w": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                 for w in (s.words or []) if w.word.strip()]
        segments.append({"start": float(s.start), "end": float(s.end),
                         "text": s.text.strip(), "words": words})
        pct = min(100, int(100 * s.end / max(duration, 1)))
        print(f"\r      {pct:3d}%", end="", flush=True)
    print()
    return segments


def fallback_segments(duration: float, step: float = 5.0):
    """No speech found (pure gameplay)? Use fixed 5s blocks so clipping still works."""
    out, t = [], 0.0
    while t < duration:
        out.append({"start": t, "end": min(t + step, duration), "text": "", "words": []})
        t += step
    return out


# ----------------------------------------------------------------------------
# Steps 3-4: score windows and pick the best clips
# ----------------------------------------------------------------------------
def score_window(segs, z, start: float, end: float):
    a = max(0, int(start))
    b = min(len(z), max(a + 1, int(math.ceil(end))))
    w = z[a:b]
    if w.size == 0:
        return -99.0, ""
    k = max(1, int(w.size * 0.2))
    peak = float(np.mean(np.sort(w)[-k:]))     # how loud are the loudest 20% of seconds
    mean = float(np.mean(w))                    # how loud overall
    text = " ".join(s["text"] for s in segs if s["text"]).strip()
    rate = len(text.split()) / max(end - start, 1.0)   # words per second
    kw = len(EXCITE_RE.findall(text))
    punct = text.count("!") + 0.5 * text.count("?")
    score = (1.0 * peak + 0.5 * mean + 0.4 * min(kw, 5)
             + 0.15 * min(punct, 6) + 0.3 * min(rate / 3.0, 1.5))
    return score, text


def pick_clips(segments, z, duration, num, min_len, max_len):
    cands = []
    n = len(segments)
    for i in range(n):
        start = segments[i]["words"][0]["start"] if segments[i]["words"] else segments[i]["start"]
        for j in range(i, n):
            end = segments[j]["end"]
            length = end - start
            if length > max_len:
                break
            if length < min_len:
                continue
            score, text = score_window(segments[i:j + 1], z, start, end)
            cands.append((score, start, end, text))

    cands.sort(key=lambda c: c[0], reverse=True)
    chosen = []
    for score, start, end, text in cands:
        if all(end + 1.0 <= c["start"] or start >= c["end"] + 1.0 for c in chosen):
            chosen.append({"score": round(score, 2),
                           "start": max(0.0, start - 0.25),
                           "end": min(duration, end + 0.4),
                           "text": text})
        if len(chosen) >= num:
            break
    return chosen


# ----------------------------------------------------------------------------
# Step 5: captions + render
# ----------------------------------------------------------------------------
def build_ass(segments, clip_start, clip_end, W, H, font, words_per_line, path: Path) -> bool:
    words = [w for s in segments for w in s["words"]
             if w["start"] >= clip_start - 0.05 and w["end"] <= clip_end + 0.05]
    if not words:
        return False

    # group words into short caption lines
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= words_per_line
                    or w["start"] - cur[-1]["end"] > 0.7
                    or cur[-1]["w"][-1:] in ".?!"):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    fs = int(min(W, H) * 0.075)
    outline = max(2, fs // 14)
    margin_v = int(H * 0.25) if H > W else int(H * 0.08)

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{fs},{BASE_COLOR},&H000000FF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},2,2,60,60,{margin_v},1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def clean(t):
        return t.replace("\\", "").replace("{", "").replace("}", "").upper()

    for gi, g in enumerate(groups):
        next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
        for i, w in enumerate(g):
            t0 = w["start"] - clip_start
            if i + 1 < len(g):
                t1 = g[i + 1]["start"] - clip_start
            else:
                t1 = w["end"] - clip_start + 0.25
                if next_start is not None:
                    t1 = min(t1, next_start - clip_start)
            t1 = max(t1, t0 + 0.08)
            parts = []
            for k, ww in enumerate(g):
                if k == i:
                    parts.append(f"{{\\1c{HIGHLIGHT_COLOR}\\fscx112\\fscy112}}{clean(ww['w'])}"
                                 f"{{\\1c{BASE_COLOR}\\fscx100\\fscy100}}")
                else:
                    parts.append(clean(ww["w"]))
            lines.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Default,,0,0,0,,{' '.join(parts)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def build_filter(W, H, layout, crop_x, ass_name):
    if layout == "crop":
        vf = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H}:(iw-ow)*{crop_x}:(ih-oh)/2,setsar=1")
    else:  # blur: whole frame visible, blurred copy fills the empty space
        bw, bh = max(4, (W // 10) // 2 * 2), max(4, (H // 10) // 2 * 2)
        vf = (f"[0:v]split=2[bg][fg];"
              f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"scale={bw}:{bh},boxblur=3:2,scale={W}:{H}[bgb];"
              f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[fgs];"
              f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1")
    if ass_name:
        vf += f",subtitles={ass_name}"
    return vf + "[v]"


def render_clip(video: Path, clip, out_dir: Path, out_name: str, ass_name, W, H, args):
    fg = build_filter(W, H, args.layout, args.crop_x, ass_name)
    if args.gpu:
        venc = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(args.crf), "-b:v", "0"]
    else:
        venc = ["-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf)]
    venc += ["-pix_fmt", "yuv420p", "-profile:v", "high"]

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{clip['start']:.3f}", "-t", f"{clip['end'] - clip['start']:.3f}",
           "-i", str(video),
           "-filter_complex", fg, "-map", "[v]", "-map", "0:a:0?",
           *venc, "-c:a", "aac", "-b:a", "256k", "-ar", "48000"]
    if not args.no_loudnorm:
        cmd += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]   # standard social-media loudness
    cmd += ["-movflags", "+faststart", out_name]
    # cwd = output folder so the subtitle file can be referenced by a plain
    # relative name (avoids Windows drive-letter escaping problems in filters)
    run(cmd, cwd=str(out_dir))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Turn a long video into captioned short clips.")
    p.add_argument("video", help="path to a video file (e.g. a stream recording)")
    p.add_argument("--out", default="clips", help="output folder (default: clips)")
    p.add_argument("--num-clips", type=int, default=5)
    p.add_argument("--min-len", type=float, default=20, help="shortest clip in seconds")
    p.add_argument("--max-len", type=float, default=55, help="longest clip in seconds")
    p.add_argument("--format", choices=list(SIZES), default="vertical")
    p.add_argument("--size", help="custom output size like 1080x1920 (overrides --format)")
    p.add_argument("--layout", choices=["blur", "crop"], default="blur",
                   help="blur = full frame + blurred background, crop = fill the frame")
    p.add_argument("--crop-x", type=float, default=0.5,
                   help="for --layout crop: 0 = left edge, 0.5 = centre, 1 = right edge")
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--font", default="Arial", help='caption font (Hindi on Windows: "Nirmala UI")')
    p.add_argument("--words-per-line", type=int, default=3)
    p.add_argument("--model", default="small", help="whisper model: tiny, base, small, medium, large-v3")
    p.add_argument("--language", default=None, help="e.g. en, hi (default: auto-detect)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="where Whisper runs")
    p.add_argument("--transcript", help="use an existing transcript JSON instead of running Whisper")
    p.add_argument("--retranscribe", action="store_true", help="ignore the cached transcript")
    p.add_argument("--gpu", action="store_true", help="encode video with NVIDIA NVENC (faster)")
    p.add_argument("--crf", type=int, default=16, help="quality: lower = better/larger (16 is near-lossless)")
    p.add_argument("--preset", default="medium", help="x264 speed preset: ultrafast ... slow")
    p.add_argument("--no-loudnorm", action="store_true", help="keep original audio loudness")
    return p.parse_args()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    check_tools()

    video = Path(args.video).resolve()
    if not video.exists():
        sys.exit(f"File not found: {video}")
    out_dir = Path(args.out).resolve()
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    stem = safe_name(video.stem)

    if args.size:
        m = re.fullmatch(r"(\d+)x(\d+)", args.size.lower())
        if not m:
            sys.exit("--size must look like 1080x1920")
        W, H = int(m.group(1)), int(m.group(2))
    else:
        W, H = SIZES[args.format]
    W, H = W - W % 2, H - H % 2

    duration = probe_duration(video)
    print(f"Video: {video.name}  ({fmt_clock(duration)})  ->  {W}x{H}")

    # 1. audio
    print("[1/5] Extracting audio and measuring loudness...")
    wav = work / f"{stem}.16k.wav"
    has_audio = extract_audio(video, wav)
    z = loudness_z(wav if has_audio else None, duration)

    # 2. transcript
    if args.transcript:
        segments = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        print(f"[2/5] Loaded transcript: {args.transcript}")
    else:
        cache = work / f"{stem}.{args.model}.transcript.json"
        if cache.exists() and not args.retranscribe:
            segments = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[2/5] Using cached transcript ({cache.name})")
        elif has_audio:
            segments = transcribe(wav, duration, args.model, args.language, args.device)
            cache.write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            segments = []
    if not segments:
        print("      No speech found - falling back to loudness-only detection.")
        segments = fallback_segments(duration)

    # 3-4. choose clips
    print("[3/5] Scoring candidate moments...")
    clips = pick_clips(segments, z, duration, args.num_clips, args.min_len, args.max_len)
    if not clips:
        sys.exit("No clips found. Try a lower --min-len or a longer video.")
    print(f"[4/5] Picked {len(clips)} clips:")
    for i, c in enumerate(clips, 1):
        preview = (c["text"][:70] + "...") if len(c["text"]) > 70 else c["text"]
        print(f"      #{i}  {fmt_clock(c['start'])} - {fmt_clock(c['end'])}  score {c['score']:>5}  {preview}")

    # 5. render
    print("[5/5] Rendering clips (this is the slow part)...")
    results = []
    for i, c in enumerate(clips, 1):
        name = f"{stem}_clip{i:02d}"
        ass_name = None
        if not args.no_captions:
            ass_path = out_dir / f"{name}.ass"
            if build_ass(segments, c["start"], c["end"], W, H, args.font, args.words_per_line, ass_path):
                ass_name = ass_path.name
        render_clip(video, c, out_dir, f"{name}.mp4", ass_name, W, H, args)
        print(f"      done: {name}.mp4")
        results.append({"file": f"{name}.mp4", **c})

    (out_dir / f"{stem}_clips.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        wav.unlink()
    except OSError:
        pass
    print(f"\nAll done. Clips are in: {out_dir}")


if __name__ == "__main__":
    main()
