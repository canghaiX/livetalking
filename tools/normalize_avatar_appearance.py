"""Match presenter foreground exposure to the canonical smile action.

The correction is deliberately conservative: it changes low-frequency LAB
brightness and a small chroma offset inside a feathered presenter mask. The
background and facial detail remain untouched, so the resulting clip can be
used by the existing Wav2Lip avatar generator without changing geometry.
"""

import argparse
import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def edge_color(frame):
    h, w = frame.shape[:2]
    band = max(3, int(w * .035))
    samples = np.concatenate([
        frame[:int(h * .08)].reshape(-1, 3),
        frame[:, :band].reshape(-1, 3),
        frame[:, -band:].reshape(-1, 3),
    ])
    return np.median(samples, axis=0).astype(np.float32)


def soft_subject_mask(frame, background=None, feather=20):
    """Estimate a feathered person mask from the gray studio edge field."""
    background = edge_color(frame) if background is None else np.asarray(background, dtype=np.float32)
    distance = np.linalg.norm(frame.astype(np.float32) - background, axis=2)
    mask = (distance > 18).astype(np.uint8) * 255
    h, w = mask.shape
    # Keep the center connected while excluding isolated compression noise.
    central = np.zeros_like(mask)
    central[int(h * .04):, int(w * .12):int(w * .88)] = 255
    mask = cv2.bitwise_and(mask, central)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, float(feather) / 3.0)).astype(np.float32) / 255.0


def inner_face_mask(shape, box, feather=18):
    """Return a soft inner face mask, avoiding the detector-box boundary."""
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return mask.astype(np.float32)
    pad_x = max(2, int((x2 - x1) * .12))
    pad_top = max(2, int((y2 - y1) * .08))
    pad_bottom = max(2, int((y2 - y1) * .06))
    cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2),
                (max(2, (x2 - x1 - 2 * pad_x) // 2),
                 max(2, (y2 - y1 - pad_top - pad_bottom) // 2)),
                0, 0, 360, 255, -1)
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, float(feather) / 3.0)).astype(np.float32) / 255.0


def _face_values(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), dtype=np.float32)
    crop = frame[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Exclude the outer detector margin where gray background and hair are
    # most likely to dominate the statistics.
    mx, my = max(1, int(crop.shape[1] * .12)), max(1, int(crop.shape[0] * .1))
    return lab[my:max(my + 1, crop.shape[0] - my),
               mx:max(mx + 1, crop.shape[1] - mx)].reshape(-1, 3)


def appearance_profile(frames, boxes):
    values = [_face_values(frame, box) for frame, box in zip(frames, boxes)]
    values = [item for item in values if len(item)]
    if not values:
        raise ValueError('无法从素材计算人脸光照统计')
    pixels = np.concatenate(values, axis=0)
    luminance = pixels[:, 0]
    p10, p90 = np.percentile(luminance, [10, 90])
    return {
        'l_mean': float(np.median(luminance)),
        'l_p10': float(p10),
        'l_p90': float(p90),
        'a_mean': float(np.median(pixels[:, 1])),
        'b_mean': float(np.median(pixels[:, 2])),
    }


def correction_from_profiles(source, target):
    source_span = max(1.0, source['l_p90'] - source['l_p10'])
    target_span = max(1.0, target['l_p90'] - target['l_p10'])
    # Conservative limits prevent a noisy clip from becoming unnaturally flat
    # or overexposed while still removing the visible action-to-action jump.
    gain = float(np.clip(target_span / source_span, .94, 1.06))
    offset = float(np.clip(target['l_mean'] - gain * source['l_mean'], -8.0, 8.0))
    return {
        'l_gain': gain,
        'l_offset': offset,
        'a_offset': float(np.clip(target['a_mean'] - source['a_mean'], -3.0, 3.0)),
        'b_offset': float(np.clip(target['b_mean'] - source['b_mean'], -3.0, 3.0)),
    }


def apply_appearance_correction(frame, correction, subject_mask, face_mask=None):
    """Apply a low-frequency foreground-only LAB correction."""
    mask = np.clip(np.asarray(subject_mask, dtype=np.float32), 0.0, 1.0)
    if face_mask is not None:
        # Face correction is allowed to reach the detector box only through a
        # feathered mask; the wider subject mask handles hair and clothing.
        mask = np.maximum(mask, np.asarray(face_mask, dtype=np.float32))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    corrected = lab.copy()
    corrected[..., 0] = np.clip(
        corrected[..., 0] * correction['l_gain'] + correction['l_offset'], 0, 255)
    corrected[..., 1] = np.clip(corrected[..., 1] + correction['a_offset'], 0, 255)
    corrected[..., 2] = np.clip(corrected[..., 2] + correction['b_offset'], 0, 255)
    corrected = cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    output = frame.astype(np.float32) * (1.0 - mask[..., None]) + corrected * mask[..., None]
    return np.clip(output, 0, 255).astype(np.uint8)


def _read_sample_frames(path, limit=None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while limit is None or len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f'无法读取视频: {path}')
    return frames


def normalize_video(input_path, output_path, target_profile, source_boxes,
                    target_background=None, report_frames=None):
    """Normalize one aligned clip and return measurable correction metadata."""
    input_path, output_path = Path(input_path), Path(output_path)
    first = _read_sample_frames(input_path, 1)[0]
    height, width = first.shape[:2]
    if (width, height) != (786, 1178):
        raise ValueError(f'输入必须为786×1178，实际为{width}×{height}')
    samples = _read_sample_frames(input_path, report_frames or 25)
    source_profile = appearance_profile(samples, source_boxes[:len(samples)])
    correction = correction_from_profiles(source_profile, target_profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '786x1178', '-r', '25',
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output_path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(input_path))
    count = 0
    edge_errors = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count >= len(source_boxes):
                raise ValueError('人脸框数量少于视频帧数')
            # Estimate the matte from the source frame.  Using the target gray
            # as the segmentation color would make a darker clip look like a
            # foreground object and create a halo at the silhouette.
            subject = soft_subject_mask(frame, edge_color(frame))
            if target_background is not None:
                # Match the flat studio field before the foreground grade. The
                # feathered inverse mask keeps hair, hands and jacket edges
                # from acquiring a hard gray outline.
                background_delta = np.asarray(target_background, dtype=np.float32) - edge_color(frame)
                frame = np.clip(
                    frame.astype(np.float32) + background_delta * (1.0 - subject[..., None]),
                    0, 255).astype(np.uint8)
            face = inner_face_mask(frame.shape, source_boxes[count])
            corrected = apply_appearance_correction(frame, correction, subject, face)
            encoder.stdin.write(corrected.tobytes())
            edge = np.concatenate([corrected[:int(height * .08)].reshape(-1, 3),
                                   corrected[:, :int(width * .035)].reshape(-1, 3),
                                   corrected[:, -int(width * .035):].reshape(-1, 3)])
            if target_background is not None:
                edge_errors.append(float(np.abs(edge.mean(axis=0) - target_background).mean()))
            count += 1
    finally:
        cap.release()
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f'ffmpeg 编码失败，退出码 {code}')
    return {'width': width, 'height': height, 'fps': 25,
            'frame_count': count, 'source_profile': source_profile,
            'target_profile': target_profile, 'correction': correction,
            'edge_mae': float(np.mean(edge_errors)) if edge_errors else None}


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-boxes', required=True, type=Path)
    parser.add_argument('--source-boxes', required=True, type=Path)
    args = parser.parse_args()
    target_boxes = pickle.loads(args.reference_boxes.read_bytes())
    source_boxes = pickle.loads(args.source_boxes.read_bytes())
    reference = _read_sample_frames(args.reference, 101)
    target = appearance_profile(reference, target_boxes[:len(reference)])
    report = normalize_video(args.input, args.output, target, source_boxes)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    _main()
