"""Build versioned, composition/appearance-normalized railway presenter avatars.

The command is deliberately transactional: all four actions are generated in a
temporary staging area and the package is published only after every action has
passed video, face and Avatar completeness checks.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from avatars.wav2lip.face_detection.detection.sfd.sfd_detector import SFDDetector
from avatars.wav2lip.genavatar import generate_avatar
from server.avatar_video_validation import probe_video, validate_faces
from tools.align_avatar_video import align_video, box_stats
from tools.normalize_avatar_appearance import (
    _read_sample_frames,
    appearance_profile,
    edge_color,
    normalize_video,
)
from tools.prepare_avatar_action import match_background


PACKAGE_ID = "railway_presenter_v2"
TARGET_SIZE = (786, 1178)
REFERENCE_ACTION = "welcome_smile_facefix_v2"
ACTIONS = {
    "welcome_smile_uniform_v3": ("welcome_smile_facefix_v2", "欢迎微笑（构图光影统一）", 1.0),
    "neutral_explain_uniform_v3": ("neutral_explain_facefix_v2", "快速抬手介绍（构图光影统一）", 0.66),
    "gentle_introduce_07_uniform_v3": ("gentle_introduce_07_facefix_v2", "温和介绍（构图光影统一）", 1.0),
    "closing_thanks_uniform_v3": ("closing_thanks_facefix_v2", "总结致意（构图光影统一）", 1.0),
}


def _run_ffmpeg(source: Path, output: Path, speed: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = ["fps=25"]
    if speed != 1.0:
        filters.insert(0, f"setpts={speed}*PTS")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-vf", ",".join(filters), "-an", "-r", "25",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _avatar_complete(folder: Path, frame_count: int) -> None:
    coords = folder / "coords.pkl"
    if not coords.is_file() or not (folder / "full_imgs").is_dir() or not (folder / "face_imgs").is_dir():
        raise ValueError(f"Avatar 输出不完整: {folder}")
    with coords.open("rb") as stream:
        if len(pickle.load(stream)) != frame_count:
            raise ValueError(f"coords.pkl 帧数不完整: {folder}")
    for kind in ("full_imgs", "face_imgs"):
        if len(list((folder / kind).glob("*.png"))) != frame_count:
            raise ValueError(f"{kind} 帧数不完整: {folder}")


def _stats(boxes):
    center_x, center_y, width, height = box_stats(boxes)
    return {"center_x": center_x, "center_y": center_y, "width": width, "height": height}


def _validate_processed_faces(path: Path, report: dict, detector, expected_boxes):
    """Validate post-color frames while rejecting background-only SFD false positives."""
    cap = cv2.VideoCapture(str(path))
    boxes = []
    missing = []
    batch = []
    try:
        def detect_batch():
            if not batch:
                return
            detected = detector.detect_from_batch(np.asarray(batch)[..., ::-1].copy())
            for frame, faces, expected in zip(batch, detected, expected_boxes[len(boxes):]):
                if not faces:
                    missing.append(len(boxes))
                    boxes.append(list(expected))
                    continue
                ex = np.asarray(expected, dtype=np.float32)
                def score(face):
                    candidate = np.asarray(face[:4], dtype=np.float32)
                    ix1, iy1 = np.maximum(candidate[:2], ex[:2])
                    ix2, iy2 = np.minimum(candidate[2:], ex[2:])
                    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area = max(1.0, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
                    return intersection / area
                chosen = max(faces, key=score)[:4]
                x1, y1, x2, y2 = [int(v) for v in chosen]
                boxes.append([max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)])
            batch.clear()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            batch.append(frame)
            if len(batch) == 4:
                detect_batch()
        detect_batch()
    finally:
        cap.release()
    if len(boxes) != report["frame_count"]:
        raise ValueError(f"处理后视频帧数不完整: {path}")
    if len(missing) > max(0, round(report["frame_count"] * .02)):
        raise ValueError(f"处理后人脸覆盖率低于98%: {path}")
    if any((box[2] - box[0]) < 80 for box in boxes):
        raise ValueError(f"处理后存在过小人脸: {path}")
    report.update(face_coverage=round(1 - len(missing) / max(1, len(boxes)), 4),
                  missing_frames=missing, multiple_face_frames=[], small_face_frames=[],
                  repaired_frames=missing)
    return boxes


def build(root: Path, weights: Path, package_id: str = PACKAGE_ID) -> dict:
    package_root = root / "data" / "avatar_packages" / package_id
    avatar_root = root / "data" / "avatars"
    source_root = package_root / "source"
    reference_video = source_root / f"{REFERENCE_ACTION}.mp4"
    if not reference_video.is_file():
        raise FileNotFoundError("缺少 01 facefix_v2 参考视频")
    for avatar_id in ACTIONS:
        if (avatar_root / f"{package_id}__{avatar_id}").exists():
            raise FileExistsError(f"拒绝覆盖已有 Avatar: {avatar_id}")

    detector = SFDDetector("cuda", str(weights))
    reference_report = probe_video(reference_video, TARGET_SIZE)
    reference_sfd_boxes = validate_faces(reference_video, reference_report, detector,
                                         min_face_width=80)
    reference_boxes = reference_sfd_boxes[:101]
    target = box_stats(reference_boxes)
    reference_frames = _read_sample_frames(reference_video, 101)
    target_profile = appearance_profile(reference_frames, reference_boxes)
    target_background = edge_color(reference_frames[0])
    temporary_root = Path(tempfile.mkdtemp(prefix=".uniform-v3-", dir=str(package_root)))
    avatar_staging = Path(tempfile.mkdtemp(prefix=".avatars-uniform-v3-", dir=str(avatar_root)))
    reports = {}
    try:
        for avatar_id, (source_action, label, speed) in ACTIONS.items():
            source_video = source_root / f"{source_action}.mp4"
            if not source_video.is_file():
                raise FileNotFoundError(f"缺少动作源视频: {source_video}")
            work = temporary_root / avatar_id
            work.mkdir(parents=True, exist_ok=True)
            cfr = work / "cfr25.mp4"
            aligned = work / "aligned.mp4"
            bgfixed = work / "background_fixed.mp4"
            final_video = work / "uniform_v3.mp4"
            _run_ffmpeg(source_video, cfr, speed)
            cfr_report = probe_video(cfr, TARGET_SIZE)
            cfr_boxes = validate_faces(cfr, cfr_report, detector, min_face_width=80)
            alignment = align_video(cfr, aligned, target, detector, TARGET_SIZE)
            aligned_report = probe_video(aligned, TARGET_SIZE)
            aligned_boxes = validate_faces(aligned, aligned_report, detector, min_face_width=80)
            background = match_background(aligned, bgfixed, reference_video)
            bg_report = probe_video(bgfixed, TARGET_SIZE)
            bg_boxes = _validate_processed_faces(bgfixed, bg_report, detector, aligned_boxes)
            appearance = normalize_video(
                bgfixed, final_video, target_profile, bg_boxes,
                target_background=target_background,
            )
            final_report = probe_video(final_video, TARGET_SIZE)
            final_boxes = _validate_processed_faces(final_video, final_report, detector, aligned_boxes)
            measured = _stats(final_boxes[:min(101, len(final_boxes))])
            if abs(measured["center_x"] - target[0]) > 4 or abs(measured["center_y"] - target[1]) > 4:
                raise ValueError(f"{avatar_id} 人脸中心偏差超过4px: {measured} vs {target}")
            if abs(measured["width"] - target[2]) > 4:
                raise ValueError(f"{avatar_id} 人脸宽度偏差超过4px: {measured} vs {target}")
            if avatar_id == "neutral_explain_uniform_v3" and not 155 <= final_report["frame_count"] <= 175:
                raise ValueError(f"03加速帧数异常: {final_report['frame_count']}")
            with tempfile.TemporaryDirectory(prefix="generated-", dir=avatar_staging) as generated_dir:
                generate_avatar(str(final_video), avatar_id, generated_dir, img_size=256,
                                pads=[0, 10, 0, 0], face_det_batch_size=4,
                                validated_boxes=final_boxes)
                generated = Path(generated_dir) / avatar_id
                _avatar_complete(generated, final_report["frame_count"])
                staged_avatar = avatar_staging / f"{package_id}__{avatar_id}"
                os.replace(generated, staged_avatar)
            reports[avatar_id] = {
                "id": avatar_id, "label": label,
                "avatar_id": f"{package_id}__{avatar_id}", "status": "completed",
                "progress": 100, "error": "", "validation": final_report,
                "alignment": alignment, "measured_face": measured,
                "appearance": appearance, "background": background,
                "source": str(source_video), "task_id": "uniform-v3",
            }

        # Publish only after every generated directory and report has passed.
        rollback = package_root / "rollback-uniform-v3"
        rollback.mkdir(exist_ok=True)
        for name in ("package.json", "manifest.json", "active.json"):
            original = package_root / name
            if original.exists():
                shutil.copy2(original, rollback / name)
        for staged in sorted(avatar_staging.glob(f"{package_id}__*")):
            os.replace(staged, avatar_root / staged.name)
        for avatar_id in ACTIONS:
            final_video = temporary_root / avatar_id / "uniform_v3.mp4"
            os.replace(final_video, source_root / f"{avatar_id}.mp4")

        package = json.loads((package_root / "package.json").read_text(encoding="utf-8"))
        existing = {item["id"]: item for item in package.get("actions", [])}
        existing.update(reports)
        package["actions"] = list(existing.values())
        package["default_action"] = "welcome_smile_uniform_v3"
        package["published"] = True
        manifest = dict(package)
        _atomic_json(package_root / "package.json", package)
        _atomic_json(package_root / "manifest.json", manifest)
        _atomic_json(package_root / "active.json", {
            "package_id": package_id,
            "avatar_id": f"{package_id}__welcome_smile_uniform_v3",
        })
        return {"actions": reports, "target_face": _stats(reference_boxes), "rollback": str(rollback)}
    except Exception:
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
        shutil.rmtree(avatar_staging, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--weights", type=Path, default=Path("models/s3fd.pth"))
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.weights.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
