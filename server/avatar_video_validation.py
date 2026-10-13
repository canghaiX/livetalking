"""Strict, bounded video validation for Wav2Lip character packages."""
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class VideoValidationError(ValueError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def probe_video(path, expected_size=None):
    if Path(path).suffix.lower() != '.mp4':
        raise VideoValidationError('只接受 MP4 文件')
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_streams', '-show_format',
            '-show_frames', '-show_entries',
            'frame=media_type,best_effort_timestamp_time:stream:format',
            '-of', 'json', str(path),
        ], check=True, capture_output=True, text=True, timeout=60)
        metadata = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        raise VideoValidationError('无法读取视频，请检查文件是否损坏') from exc
    streams = [s for s in metadata.get('streams', []) if s['codec_type'] == 'video']
    if len(streams) != 1:
        raise VideoValidationError('视频必须且只能包含一个视频流')
    stream = streams[0]
    if stream.get('codec_name') != 'h264' or 'mp4' not in metadata['format']['format_name'].split(','):
        raise VideoValidationError('视频必须使用 MP4 容器和 H.264 编码')
    width, height = stream['width'], stream['height']
    if min(width, height) < 720 or max(width, height) > 1920 or width * height > 1920 * 1080:
        raise VideoValidationError('视频尺寸应在 720p 到 1080p 范围内')
    if expected_size and (width, height) != tuple(expected_size):
        raise VideoValidationError(f'动作尺寸必须与角色首段视频一致：{expected_size[0]}×{expected_size[1]}')
    if any(float(s.get('rotation', 0)) != 0 for s in stream.get('side_data_list', [])) or float(stream.get('tags', {}).get('rotate', 0)) != 0:
        raise VideoValidationError('请先导出无旋转标记的视频')
    try:
        fps = Fraction(stream['avg_frame_rate'])
        nominal_fps = Fraction(stream['r_frame_rate'])
        duration = float(stream.get('duration', metadata['format'].get('duration', 0)))
        timestamps = [float(f['best_effort_timestamp_time']) for f in metadata['frames'] if f.get('media_type') == 'video']
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        raise VideoValidationError('视频缺少可靠的帧率或时间戳') from exc
    if fps != 25 or nominal_fps != 25 or len(timestamps) < 2 or any(
        abs(b - a - 0.04) > 0.0002 for a, b in zip(timestamps, timestamps[1:])
    ):
        raise VideoValidationError('只接受恒定 25fps 视频；请先导出为 25fps')
    if not math.isfinite(duration) or not 3 <= duration <= 15 or not 75 <= len(timestamps) <= 375:
        raise VideoValidationError('视频时长必须在 3–15 秒之间')
    return {'width': width, 'height': height, 'fps': 25, 'duration': duration,
            'frame_count': len(timestamps), 'codec': 'h264'}


def validate_faces(path, report, detector, progress=lambda _: None):
    """Keep every detected box; never substitute the whole image for a missing face."""
    cap = cv2.VideoCapture(str(path))
    boxes, missing, multiple, small = [], [], [], []
    batch = []

    def detect_batch():
        if not batch:
            return
        detected = detector.detect_from_batch(np.asarray(batch)[..., ::-1].copy())
        if len(detected) != len(batch):
            raise VideoValidationError('人脸检测器返回了不完整的结果', report)
        for frame, faces in zip(batch, detected):
            index = len(boxes)
            if len(faces) > 1:
                multiple.append(index)
            if not faces:
                missing.append(index)
                boxes.append(None)
                continue
            x1, y1, x2, y2 = faces[0][:4]
            h, w = frame.shape[:2]
            box = [max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))]
            if box[2] - box[0] < 160 or box[3] <= box[1]:
                small.append(index)
            boxes.append(box)
        progress(min(65, 10 + int(len(boxes) / report['frame_count'] * 55)))
        batch.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if len(boxes) + len(batch) >= 375:
                raise VideoValidationError('视频帧数超过限制', report)
            if frame.shape[:2] != (report['height'], report['width']):
                raise VideoValidationError('解码尺寸与视频元数据不一致', report)
            batch.append(frame)
            if len(batch) == 4:
                detect_batch()
        detect_batch()
    finally:
        cap.release()
    coverage = (len(boxes) - len(missing)) / max(1, len(boxes))
    report.update(face_coverage=round(coverage, 4), missing_frames=missing,
                  multiple_face_frames=multiple, small_face_frames=small)
    if len(boxes) != report['frame_count']:
        raise VideoValidationError('无法完整解码视频帧', report)
    if multiple:
        raise VideoValidationError(f'第 {multiple[0] + 1} 帧检测到多个人脸', report)
    if small:
        raise VideoValidationError(f'第 {small[0] + 1} 帧人脸宽度不足 160 像素', report)
    if coverage < .98:
        raise VideoValidationError(f'人脸检测覆盖率 {coverage:.1%}，低于 98%；第 {missing[0] + 1} 帧漏检', report)
    # Very brief misses may be repaired by adjacent boxes; larger gaps require new footage.
    if any(all(boxes[i + j] is None for j in range(3)) for i in range(len(boxes) - 2)):
        raise VideoValidationError('连续三帧未检测到人脸，请重新提供素材', report)
    known = [i for i, box in enumerate(boxes) if box is not None]
    if not known:
        raise VideoValidationError('视频中未检测到人脸', report)
    for i in missing:
        boxes[i] = boxes[min(known, key=lambda j: abs(j - i))].copy()
    report['repaired_frames'] = missing
    return boxes
