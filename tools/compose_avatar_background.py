"""Replace an aligned studio background while preserving the presenter.

The source and reference clips are already in the same 786x1178 geometry. A
small GrabCut matte is used for the moving presenter, while the reference
plate is built from a clean gray edge field. The output is intentionally
ordinary H.264 so it can be fed to the existing Wav2Lip avatar generator.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _read_frame(path, index=0):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f'无法读取视频帧: {path}')
    return frame


def _edge_color(frame):
    h, w = frame.shape[:2]
    band = max(4, int(w * .035))
    regions = [frame[:int(h * .08)], frame[:, :band], frame[:, -band:]]
    values = np.concatenate([part.reshape(-1, 3) for part in regions], axis=0)
    return np.median(values, axis=0).astype(np.float32)


def _subject_mask(frame):
    """Return a feathered temporal-friendly presenter matte."""
    h, w = frame.shape[:2]
    border = _edge_color(frame)
    distance = np.linalg.norm(frame.astype(np.float32) - border, axis=2)
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[:20] = cv2.GC_BGD
    mask[-12:] = cv2.GC_BGD
    mask[:, :20] = cv2.GC_BGD
    mask[:, -20:] = cv2.GC_BGD
    # Dark hair/jacket and skin are reliable foreground seeds on this gray set.
    mask[distance > 32] = cv2.GC_PR_FGD
    central = np.zeros((h, w), dtype=np.uint8)
    central[int(h * .06):, int(w * .08):int(w * .92)] = 1
    mask[(distance > 50) & (central == 1)] = cv2.GC_FGD
    # Keep the presenter connected even when a face frame is close to gray.
    mask[int(h * .08):int(h * .99), int(w * .22):int(w * .78)] = np.maximum(
        mask[int(h * .08):int(h * .99), int(w * .22):int(w * .78)], cv2.GC_PR_FGD)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(frame, mask, None, bg_model, fg_model, 2, cv2.GC_INIT_WITH_MASK)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.GaussianBlur(binary, (0, 0), 2.0).astype(np.float32) / 255.0


def _background_plate(reference):
    """Inpaint the presenter from a canonical frame to preserve gray texture."""
    source = _read_frame(reference)
    color = _edge_color(source)
    matte = (_subject_mask(source) > .08).astype(np.uint8) * 255
    # Dilate beyond hair and hands before inpainting so no presenter pixels
    # become part of the background plate. A light blur retains the studio's
    # broad illumination while removing compression noise.
    matte = cv2.dilate(matte, np.ones((31, 31), np.uint8), iterations=1)
    plate = cv2.inpaint(source, matte, 15, cv2.INPAINT_TELEA)
    plate = cv2.GaussianBlur(plate, (0, 0), 3)
    return plate.astype(np.uint8), color


def compose(input_path, output_path, reference):
    plate, target_color = _background_plate(reference)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f'无法打开视频: {input_path}')
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (786, 1178):
        raise ValueError('输入必须先对齐为786x1178')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', '786x1178', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264',
               '-crf', '18', '-pix_fmt', 'yuv420p', str(output_path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    errors = []
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            matte = _subject_mask(frame)
            composed = frame.astype(np.float32) * matte[..., None] + plate * (1.0 - matte[..., None])
            composed = np.clip(composed, 0, 255).astype(np.uint8)
            encoder.stdin.write(composed.tobytes())
            edge = np.concatenate((composed[:int(height * .06)].reshape(-1, 3),
                                   composed[:, :int(width * .03)].reshape(-1, 3),
                                   composed[:, -int(width * .03):].reshape(-1, 3)))
            errors.append(float(np.abs(edge.mean(axis=0) - target_color).mean()))
            frames += 1
    finally:
        capture.release()
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f'ffmpeg 编码失败: {code}')
    return {'width': width, 'height': height, 'fps': 25, 'frame_count': frames,
            'target_bgr': np.round(target_color, 3).tolist(),
            'edge_mae': float(np.mean(errors)) if errors else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compose(args.input, args.output, args.reference), ensure_ascii=False))
