"""Align an avatar source video to the canonical railway-presenter framing.

The transform is estimated once per clip from the median SFD face box. A fixed
transform preserves the clip's natural motion without introducing per-frame
scale jitter.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np

from avatars.wav2lip.face_detection.detection.sfd.sfd_detector import SFDDetector
from server.avatar_video_validation import probe_video, validate_faces


def box_stats(boxes):
    values = []
    for x1, y1, x2, y2 in boxes:
        values.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    if not values:
        raise ValueError('没有可用于对齐的人脸框')
    return tuple(statistics.median(item[index] for item in values) for index in range(4))


def reference_target(reference_boxes):
    stats = [box_stats(boxes) for boxes in reference_boxes]
    return tuple(statistics.median(item[index] for item in stats) for index in range(4))


def alignment_transform(source, target):
    source_x, source_y, source_width, _ = source
    target_x, target_y, target_width, _ = target
    scale = target_width / source_width
    matrix = np.array([[scale, 0.0, target_x - scale * source_x],
                       [0.0, scale, target_y - scale * source_y]], dtype=np.float32)
    return matrix, scale


def align_video(input_path, output_path, target, detector, size=(786, 1178)):
    report = probe_video(input_path)
    # Input geometry may need enlargement; final output still uses the strict
    # 160px publishing threshold in the caller.
    boxes = validate_faces(input_path, report, detector, min_face_width=80)
    source = box_stats(boxes)
    matrix, scale = alignment_transform(source, target)
    width, height = size

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f'无法打开视频: {input_path}')
    first = None
    try:
        ok, first = cap.read()
        if not ok:
            raise ValueError(f'视频没有可解码帧: {input_path}')
        border = tuple(int(value) for value in first[0, 0])
        command = [
            'ffmpeg', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', '25', '-i', '-',
            '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(output_path),
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            frame = first
            while True:
                aligned = cv2.warpAffine(frame, matrix, (width, height),
                                         flags=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=border)
                encoder.stdin.write(aligned.tobytes())
                ok, frame = cap.read()
                if not ok:
                    break
        finally:
            encoder.stdin.close()
            return_code = encoder.wait()
        if return_code != 0:
            raise RuntimeError(f'ffmpeg 编码失败，退出码 {return_code}')
    finally:
        cap.release()
    return {'source': source, 'target': target, 'scale': scale,
            'width': width, 'height': height, 'frame_count': report['frame_count']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', action='append', required=True, type=Path,
                        help='稳定构图参考视频，可重复传入')
    parser.add_argument('--reference-window', action='append', type=str,
                        help='每个参考视频的稳定帧区间 start:end，可重复传入')
    parser.add_argument('--weights', default='models/s3fd.pth', type=Path)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f'缺少 SFD 权重: {args.weights}')
    detector = SFDDetector('cuda', str(args.weights))
    reference_boxes = []
    windows = args.reference_window or []
    if windows and len(windows) != len(args.reference):
        raise SystemExit('--reference-window 必须与 --reference 一一对应')
    for index, path in enumerate(args.reference):
        report = probe_video(path, (786, 1178))
        boxes = validate_faces(path, report, detector)
        if windows:
            start, end = (int(value) for value in windows[index].split(':', 1))
            boxes = boxes[start:end + 1]
            if not boxes:
                raise SystemExit(f'参考帧区间为空: {path} {windows[index]}')
        reference_boxes.append(boxes)
    target = reference_target(reference_boxes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = align_video(args.input, args.output, target, detector)
    output_report = probe_video(args.output, (786, 1178))
    validate_faces(args.output, output_report, detector)
    result['output_report'] = output_report
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
