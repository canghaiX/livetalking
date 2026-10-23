"""Normalize the simple gray studio background of an aligned avatar clip.

The source clips use the same gray studio setup but have a small exposure
offset.  A single global color offset keeps the correction spatially uniform,
which avoids the halo that is produced when a different correction is applied
on each side of the person's silhouette.  The offset is neutral exposure
correction, so it does not alter facial hue or geometry.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def background_samples(frame):
    height, width = frame.shape[:2]
    regions = [
        frame[:int(height * .16), :],
        # Keep the side samples outside the shoulders.  The wider strips used
        # by the first implementation included sleeves and made a good
        # background correction look numerically worse than it was.
        frame[:, :int(width * .06)],
        frame[:, int(width * .94):],
    ]
    return np.concatenate([region.reshape(-1, 3) for region in regions], axis=0)


def collect_frames(path, sample_frames=25):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        for _ in range(sample_frames):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f'无法读取背景参考视频: {path}')
    return np.asarray(frames, dtype=np.uint8)


def foreground_mask(background_frame, threshold=16.0):
    """Return a broad, feathered estimate of the stable subject silhouette."""
    # A heavily blurred frame is a good approximation of the uniform studio
    # background.  The generous dilation/feather keeps any correction seam well
    # away from hair, hands, and jacket edges.
    background_field = cv2.GaussianBlur(background_frame, (0, 0), 35)
    distance = np.linalg.norm(background_frame.astype(np.float32) - background_field, axis=2)
    foreground = (distance > threshold).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.dilate(foreground, np.ones((41, 41), np.uint8), iterations=2)
    return cv2.GaussianBlur(foreground, (0, 0), 35).astype(np.float32) / 255.0


def normalize_video(input_path, output_path, references, threshold=16.0):
    reference_frames = [collect_frames(path) for path in references]
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f'无法打开输入视频: {input_path}')
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
        '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    source_frames = collect_frames(input_path, sample_frames=25)
    target_background = np.mean([
        np.mean([background_samples(frame).mean(axis=0) for frame in frames], axis=0)
        for frames in reference_frames
    ], axis=0)
    source_background = np.mean([
        background_samples(frame).mean(axis=0) for frame in source_frames
    ], axis=0)
    # ffmpeg's yuv420p conversion shifts decoded BGR samples by a small,
    # stable amount. Compensate before encoding so the *decoded* video, which
    # is what the browser displays, matches the light-gray reference.
    codec_compensation = np.array([3.0, 2.0, 3.0], dtype=np.float32)
    correction = target_background - source_background + codec_compensation
    # The correction is a neutral exposure shift (the three channels move by
    # nearly the same amount).  Applying that uniform shift across the frame
    # is less visible than a subject-shaped matte and, crucially, cannot leave
    # a halo around hair or hands.  Keep the mask helper available for callers
    # that need a protected matte, but do not use a hard silhouette here.
    static_foreground = np.zeros(source_frames.shape[1:3], dtype=np.float32)
    frames = 0
    total_error = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            corrected = np.clip(frame.astype(np.float32) + correction, 0, 255)
            output = frame.astype(np.float32) * static_foreground[:, :, None] \
                + corrected * (1.0 - static_foreground[:, :, None])
            output = np.clip(output, 0, 255).astype(np.uint8)
            encoder.stdin.write(output.tobytes())
            total_error.append(np.abs(background_samples(output).mean(axis=0) - target_background).mean())
            frames += 1
    finally:
        capture.release()
        encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f'ffmpeg 编码失败，退出码 {return_code}')
    return {
        'input': str(input_path), 'output': str(output_path),
        'width': width, 'height': height, 'fps': fps, 'frame_count': frames,
        'target_background_bgr': np.round(target_background, 3).tolist(),
        'mean_background_error': float(np.mean(total_error)) if total_error else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', action='append', required=True, type=Path)
    parser.add_argument('--threshold', type=float, default=16.0)
    args = parser.parse_args()
    print(json.dumps(normalize_video(args.input, args.output, args.reference, args.threshold),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
