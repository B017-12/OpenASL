#!/usr/bin/env python3
"""
Extract clean, normalized MediaPipe Holistic pose JSON files from Google's Kaggle
"Isolated Sign Language Recognition" (asl-signs) dataset.

Dataset Attribution (CC-BY 4.0):
  Deaf Professional Arts Network (DPAN) & Georgia Institute of Technology
  https://www.kaggle.com/competitions/asl-signs

Usage:
  pip install pandas pyarrow numpy
  python3 scripts/extract_asl_signs.py --input /path/to/asl-signs

Options:
  --input PATH     Directory containing train.csv and train_landmark_files/
  --output PATH    Output directory (default: public/assets/poses)
  --max-candidates Max parquet files to evaluate per gloss (default: 40)
  --glosses LIST   Optional comma-separated subset of glosses (e.g. hello,thankyou,water)
"""

import argparse
import json
import math
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT_DIR / "public" / "assets" / "poses"

# MediaPipe Pose upper-body landmark indices we render on HTML5 <canvas>
# 0: nose, 11/12: shoulders, 13/14: elbows, 15/16: wrists
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16]

# Compact facial expression subset (eyebrows, eyes, mouth outline) from MediaPipe FaceMesh (468 pts)
FACE_BROW_LEFT = [70, 63, 105, 66, 107]
FACE_BROW_RIGHT = [336, 296, 334, 293, 300]
FACE_EYE_LEFT = [33, 159, 133, 145]
FACE_EYE_RIGHT = [362, 386, 263, 374]
FACE_MOUTH = [61, 13, 291, 14]
FACE_INDICES = FACE_BROW_LEFT + FACE_BROW_RIGHT + FACE_EYE_LEFT + FACE_EYE_RIGHT + FACE_MOUTH


def round_pt(x: float, y: float) -> Optional[List[float]]:
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
        return None
    return [round(float(x), 4), round(float(y), 4)]


def extract_sequence(pq_path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Load one asl-signs parquet file, score its quality, and normalize coordinates."""
    df = pd.read_parquet(pq_path, columns=["frame", "type", "landmark_index", "x", "y"])
    if df.empty:
        return None

    frames_sorted = sorted(df["frame"].unique())
    num_frames = len(frames_sorted)
    if num_frames < 8 or num_frames > 150:
        return None

    # Compute normalization anchor from median shoulder midpoint & width across frames
    pose_df = df[df["type"] == "pose"]
    l_sh = pose_df[pose_df["landmark_index"] == 11][["x", "y"]].dropna()
    r_sh = pose_df[pose_df["landmark_index"] == 12][["x", "y"]].dropna()
    if l_sh.empty or r_sh.empty:
        return None

    lx, ly = float(l_sh["x"].median()), float(l_sh["y"].median())
    rx, ry = float(r_sh["x"].median()), float(r_sh["y"].median())
    center_x = (lx + rx) * 0.5
    center_y = (ly + ry) * 0.5
    shoulder_w = math.hypot(lx - rx, ly - ry)
    if shoulder_w < 0.05:
        return None

    def norm_xy(x: float, y: float) -> Optional[List[float]]:
        if math.isnan(x) or math.isnan(y):
            return None
        # Normalize so shoulders span ~0.42 of canvas width and sit around y = 0.62
        nx = 0.5 + ((x - center_x) / shoulder_w) * 0.42
        ny = 0.62 + ((y - center_y) / shoulder_w) * 0.42
        return round_pt(nx, ny)

    hand_visible_frames = 0
    extracted_frames: List[Dict[str, Any]] = []

    for _, fdf in df.groupby("frame", sort=True):
        by_type = {t: grp.set_index("landmark_index") for t, grp in fdf.groupby("type")}

        def grab_points(type_name: str, indices: List[int]) -> Optional[List[Optional[List[float]]]]:
            tbl = by_type.get(type_name)
            if tbl is None:
                return None
            pts: List[Optional[List[float]]] = []
            valid = 0
            for idx in indices:
                if idx in tbl.index:
                    row = tbl.loc[idx]
                    pt = norm_xy(float(row["x"]), float(row["y"]))
                    if pt is not None:
                        valid += 1
                    pts.append(pt)
                else:
                    pts.append(None)
            return pts if valid > 0 else None

        pose_pts = grab_points("pose", POSE_INDICES)
        lh_pts = grab_points("left_hand", list(range(21)))
        rh_pts = grab_points("right_hand", list(range(21)))
        face_pts = grab_points("face", FACE_INDICES)

        if lh_pts is not None or rh_pts is not None:
            hand_visible_frames += 1

        extracted_frames.append(
            {
                "p": pose_pts,
                "lh": lh_pts,
                "rh": rh_pts,
                "f": face_pts,
            }
        )

    hand_ratio = hand_visible_frames / max(1, num_frames)
    if hand_ratio < 0.45:
        return None

    # Prefer sequences with high hand visibility and clean duration (~20-55 frames)
    duration_penalty = abs(num_frames - 32) / 60.0
    score = hand_ratio * 2.0 - duration_penalty

    return {
        "score": score,
        "frameCount": num_frames,
        "frames": extracted_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OpenASL pose JSONs from Kaggle asl-signs.")
    parser.add_argument("--input", required=True, type=pathlib.Path, help="Path to unzipped asl-signs folder")
    parser.add_argument("--output", default=DEFAULT_OUT_DIR, type=pathlib.Path, help="Output poses directory")
    parser.add_argument("--max-candidates", type=int, default=40, help="Candidate parquet files checked per gloss")
    parser.add_argument("--glosses", type=str, default="", help="Optional comma-separated gloss filter")
    args = parser.parse_args()

    train_csv = args.input / "train.csv"
    if not train_csv.exists():
        sys.exit(f"[error] Could not find {train_csv}. Pass the unzipped asl-signs folder to --input.")

    args.output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(train_csv)

    wanted = {g.strip().lower() for g in args.glosses.split(",") if g.strip()} if args.glosses else None
    grouped = meta.groupby("sign")

    exported = 0
    skipped = 0

    for sign_name, group in grouped:
        sign_lower = str(sign_name).strip().lower()
        if wanted and sign_lower not in wanted:
            continue

        gloss_key = sign_lower.upper().replace(" ", "_").replace("-", "_")
        best: Optional[Tuple[float, Dict[str, Any], int]] = None

        candidates = group.head(args.max_candidates)
        for _, row in candidates.iterrows():
            pq_file = args.input / str(row["path"])
            if not pq_file.exists():
                continue
            result = extract_sequence(pq_file)
            if result is None:
                continue
            if best is None or result["score"] > best[0]:
                best = (result["score"], result, int(row["participant_id"]))

        if best is None:
            print(f"[warn] No clean sequence found for {gloss_key}")
            skipped += 1
            continue

        _, seq, participant_id = best
        payload = {
            "gloss": gloss_key,
            "fps": 30,
            "source": "Google Isolated Sign Language Recognition (DPAN & Georgia Tech, CC-BY 4.0)",
            "participantId": participant_id,
            "frameCount": seq["frameCount"],
            "poseIndices": POSE_INDICES,
            "faceGroups": {
                "browLeft": [0, 5],
                "browRight": [5, 10],
                "eyeLeft": [10, 14],
                "eyeRight": [14, 18],
                "mouth": [18, 22],
            },
            "frames": seq["frames"],
        }

        out_path = args.output / f"{gloss_key}.json"
        out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        exported += 1
        print(f"[ok]   {gloss_key}.json ({seq['frameCount']} frames, {out_path.stat().st_size // 1024} KB)")

    print(f"\nDone! Exported {exported} pose JSONs ({skipped} skipped) to {args.output}")

    manifest_script = ROOT_DIR / "scripts" / "build_manifest.mjs"
    if manifest_script.exists():
        print("Rebuilding public/assets/manifest.json...")
        subprocess.run(["node", str(manifest_script)], check=True)


if __name__ == "__main__":
    main()
