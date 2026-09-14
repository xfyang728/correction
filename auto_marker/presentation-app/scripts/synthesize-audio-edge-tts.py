"""
synthesize-audio-edge-tts.py — read audio-segments.json and call edge-tts
to produce one mp3 per segment under public/audio/<chapter>/<step>.mp3.

Replaces the bash synthesize-audio.sh for Windows / no-mmx environments.

Usage:
  python scripts/synthesize-audio-edge-tts.py                # incremental
  python scripts/synthesize-audio-edge-tts.py --force        # overwrite all
  python scripts/synthesize-audio-edge-tts.py --voice=zh-CN-YunxiNeural
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_PATH = ROOT / "audio-segments.json"
OUT_DIR = ROOT / "public" / "audio"
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"


async def synth_one(text: str, out_path: Path, voice: str) -> None:
    """Synthesize one segment to mp3."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


async def main_async(args: argparse.Namespace) -> int:
    if not SEGMENTS_PATH.exists():
        print(f"✗ {SEGMENTS_PATH} not found. Run: npm run extract-narrations",
              file=sys.stderr)
        return 1

    with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
        segments = json.load(f)

    total = len(segments)
    synthesized = 0
    skipped = 0
    failed = 0

    for i, seg in enumerate(segments, 1):
        chapter = seg["chapter"]
        step = seg["step"]
        text = seg["text"]
        out_path = OUT_DIR / chapter / f"{step}.mp3"
        rel = f"{chapter}/{step}.mp3"

        if out_path.exists() and not args.force:
            skipped += 1
            print(f"[{i:>3}/{total}] {rel:<22} skip (exists)")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        try:
            await synth_one(text, out_path, args.voice)
            elapsed = int(time.time() - start)
            synthesized += 1
            print(f"[{i:>3}/{total}] {rel:<22} ✓ {elapsed}s")
        except Exception as e:
            failed += 1
            print(f"[{i:>3}/{total}] {rel:<22} ✗ FAILED: {e}", file=sys.stderr)

    print()
    print(f"✓ done — synthesized {synthesized}, skipped {skipped}, "
          f"failed {failed}")
    return 0 if failed == 0 else 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize narration audio via edge-tts")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing mp3 files")
    parser.add_argument("--voice", default=DEFAULT_VOICE,
                        help=f"edge-tts voice id (default: {DEFAULT_VOICE})")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
