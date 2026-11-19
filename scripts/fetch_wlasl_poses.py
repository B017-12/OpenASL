#!/usr/bin/env python3
"""
Download and extract real adult ASL MediaPipe Holistic pose sequences from HuggingFace:
  Kibalama/poseformer-sign-language (derived from WLASL / Voxel51/WLASL, 11,980 clips, ~2,000 glosses)

76-joint schema per frame (MediaPipe Holistic):
  0..24  : Face (0=nose, 1..4=eye1, 5..8=eye2, 9..11=brow1, 12..14=brow2, 15..23=mouth, 24=chin)
  25..33 : Upper body pose (25=nose, 26=L_sh, 27=R_sh, 28=L_el, 29=R_el, 30=L_wr, 31=R_wr, 32..33=hips)
  34..54 : Left hand (21 MediaPipe landmarks)
  55..75 : Right hand (21 MediaPipe landmarks)

Usage:
  uv run --with pyarrow --with numpy python3 scripts/fetch_wlasl_poses.py
  uv run --with pyarrow --with numpy python3 scripts/fetch_wlasl_poses.py --all
"""

import argparse
import json
import math
import pathlib
import subprocess
import tempfile
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT_DIR / "public" / "assets" / "poses"
CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "openasl_wlasl_cache"

HF_SHARDS = [
    f"https://huggingface.co/datasets/Kibalama/poseformer-sign-language/resolve/main/data/train-{i:05d}-of-00004.parquet"
    for i in range(4)
]

# High-frequency adult conversational ASL starter set (used unless --all is passed)
CORE_CONVERSATIONAL_GLOSSES = {
    # Pronouns & People
    "i", "me", "you", "he", "she", "we", "they", "my", "your", "name",
    "man", "woman", "person", "people", "friend", "family", "mother", "father",
    "brother", "sister", "child", "baby", "boy", "girl", "teacher", "student", "doctor",
    # Time & Tense
    "now", "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night",
    "day", "week", "month", "year", "time", "before", "after", "later", "soon",
    "always", "sometimes", "never", "past", "future", "again", "finish", "start", "wait",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # Questions & Negation
    "who", "what", "where", "when", "why", "how", "which", "not", "no", "yes", "nothing", "none",
    # Everyday Verbs
    "go", "come", "work", "study", "learn", "teach", "know", "understand", "think", "remember",
    "forget", "want", "need", "like", "love", "hate", "help", "give", "take", "get",
    "make", "do", "try", "use", "buy", "sell", "pay", "eat", "drink", "cook", "sleep",
    "wake", "walk", "run", "drive", "travel", "live", "stay", "leave", "meet", "see",
    "look", "watch", "hear", "listen", "talk", "say", "tell", "ask", "answer", "read",
    "write", "sign", "play", "open", "close", "find", "lose", "bring", "put", "call",
    "send", "show", "feel", "change", "build", "create", "fix", "break", "stop", "continue",
    # Common Nouns & Places
    "hello", "goodbye", "please", "thank you", "sorry", "excuse", "welcome",
    "home", "house", "room", "school", "college", "library", "office", "store", "hospital",
    "city", "country", "world", "car", "bus", "train", "plane", "bike",
    "book", "paper", "pen", "computer", "phone", "internet", "video", "camera", "music", "movie",
    "food", "water", "coffee", "tea", "milk", "bread", "apple", "pizza", "dinner", "lunch", "breakfast",
    "money", "price", "problem", "question", "idea", "plan", "project", "class", "test", "language",
    "english", "word", "sentence", "story", "news", "weather", "rain", "sun", "cold", "hot",
    # Adjectives & Adverbs
    "good", "bad", "hard", "easy", "big", "small", "long", "short", "fast", "slow",
    "new", "old", "young", "happy", "sad", "angry", "afraid", "tired", "sick", "fine",
    "nice", "beautiful", "funny", "smart", "important", "different", "same", "right", "wrong",
    "true", "false", "ready", "busy", "free", "full", "empty", "clean", "dirty",
    "really", "very", "more", "less", "many", "few", "all", "some", "here", "there",
    # Numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}


def is_valid_pt(pt: List[float]) -> bool:
    if not pt or len(pt) < 2:
        return False
    x, y = float(pt[0]), float(pt[1])
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
        return False
    # Zero-padded missing landmarks in PoseFormer dataset
    if abs(x) < 1e-5 and abs(y) < 1e-5:
        return False
    return True


def extract_wlasl_sequence(landmarks: List[List[List[float]]]) -> Optional[Dict[str, Any]]:
    """Score and normalize one 76-joint WLASL pose sequence."""
    if not landmarks or len(landmarks) < 8:
        return None

    # Trim leading/trailing all-zero padded frames
    active_frames = []
    for frame in landmarks:
        if len(frame) < 76:
            continue
        if is_valid_pt(frame[26]) and is_valid_pt(frame[27]):
            active_frames.append(frame)

    num_frames = len(active_frames)
    if num_frames < 8:
        return None

    # Compute median shoulder midpoint & width across active frames for uniform framing
    l_sh = np.array([[f[26][0], f[26][1]] for f in active_frames], dtype=np.float64)
    r_sh = np.array([[f[27][0], f[27][1]] for f in active_frames], dtype=np.float64)
    lx, ly = float(np.median(l_sh[:, 0])), float(np.median(l_sh[:, 1]))
    rx, ry = float(np.median(r_sh[:, 0])), float(np.median(r_sh[:, 1]))

    center_x = (lx + rx) * 0.5
    center_y = (ly + ry) * 0.5
    shoulder_w = math.hypot(lx - rx, ly - ry)
    if shoulder_w < 0.05:
        return None

    def norm_pt(pt: List[float]) -> Optional[List[float]]:
        if not is_valid_pt(pt):
            return None
        nx = 0.5 + ((float(pt[0]) - center_x) / shoulder_w) * 0.42
        ny = 0.58 + ((float(pt[1]) - center_y) / shoulder_w) * 0.42
        return [round(nx, 4), round(ny, 4)]

    # Downsample very long clips gently so every sign plays crisply in ~1.0-2.0s at 30fps
    step = 2 if num_frames > 68 else 1
    sampled = active_frames[::step]

    hand_visible = 0
    out_frames: List[Dict[str, Any]] = []

    for f in sampled:
        # Upper body: [nose, L_sh, R_sh, L_el, R_el, L_wr, R_wr]
        pose_pts = [norm_pt(f[i]) for i in (25, 26, 27, 28, 29, 30, 31)]

        lh_raw = [norm_pt(f[i]) for i in range(34, 55)]
        rh_raw = [norm_pt(f[i]) for i in range(55, 76)]
        lh_valid = sum(1 for p in lh_raw if p is not None)
        rh_valid = sum(1 for p in rh_raw if p is not None)

        lh_pts = lh_raw if lh_valid >= 12 else None
        rh_pts = rh_raw if rh_valid >= 12 else None
        if lh_pts is not None or rh_pts is not None:
            hand_visible += 1

        # Compact face (22 pts matching poseRenderer.js: browL[5], browR[5], eyeL[4], eyeR[4], mouth[4])
        brow1 = [norm_pt(f[i]) for i in (9, 10, 10, 11, 11)]
        brow2 = [norm_pt(f[i]) for i in (12, 13, 13, 14, 14)]
        eye1 = [norm_pt(f[i]) for i in (1, 3, 2, 4)]
        eye2 = [norm_pt(f[i]) for i in (5, 7, 6, 8)]
        mouth = [norm_pt(f[i]) for i in (17, 22, 18, 21)]
        face_pts = brow1 + brow2 + eye1 + eye2 + mouth

        out_frames.append({"p": pose_pts, "lh": lh_pts, "rh": rh_pts, "f": face_pts})

    hand_ratio = hand_visible / max(1, len(out_frames))
    if hand_ratio < 0.45:
        return None

    duration_penalty = abs(len(out_frames) - 34) / 60.0
    score = hand_ratio * 2.0 - duration_penalty

    return {"score": score, "frameCount": len(out_frames), "frames": out_frames}


def download_shard(url: str) -> pathlib.Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / pathlib.Path(url).name
    if dest.exists() and dest.stat().st_size > 10_000_000:
        return dest
    print(f"[download] {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "OpenASL/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch & extract adult WLASL pose JSONs from HuggingFace.")
    parser.add_argument("--output", default=DEFAULT_OUT_DIR, type=pathlib.Path)
    parser.add_argument("--all", action="store_true", help="Extract all ~2,000 WLASL glosses instead of core conversational set")
    parser.add_argument("--glosses", type=str, default="", help="Optional comma-separated gloss list")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.glosses:
        wanted: Optional[set] = {g.strip().lower() for g in args.glosses.split(",") if g.strip()}
    elif args.all:
        wanted = None
    else:
        wanted = CORE_CONVERSATIONAL_GLOSSES

    best_by_gloss: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    for url in HF_SHARDS:
        shard_path = download_shard(url)
        table = pq.read_table(shard_path, columns=["label", "landmarks"])
        labels = table["label"].to_pylist()
        landmarks_col = table["landmarks"]

        for idx, raw_label in enumerate(labels):
            lbl = str(raw_label or "").strip().lower()
            if not lbl or lbl == "unknown":
                continue
            if wanted is not None and lbl not in wanted:
                continue

            gloss_key = lbl.upper().replace(" ", "").replace("-", "_")
            seq = extract_wlasl_sequence(landmarks_col[idx].as_py())
            if seq is None:
                continue

            prev = best_by_gloss.get(gloss_key)
            if prev is None or seq["score"] > prev[0]:
                best_by_gloss[gloss_key] = (seq["score"], seq)

    exported = 0
    for gloss_key, (_, seq) in sorted(best_by_gloss.items()):
        payload = {
            "gloss": gloss_key,
            "fps": 30,
            "source": "WLASL MediaPipe Holistic Landmarks (Kibalama/poseformer-sign-language, C-UDA)",
            "frameCount": seq["frameCount"],
            "frames": seq["frames"],
        }
        out_path = args.output / f"{gloss_key}.json"
        out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        exported += 1

    print(f"[done] Exported {exported} real adult ASL pose JSONs to {args.output}")
    manifest_script = ROOT_DIR / "scripts" / "build_manifest.mjs"
    if manifest_script.exists():
        subprocess.run(["node", str(manifest_script)], check=True)


if __name__ == "__main__":
    main()
