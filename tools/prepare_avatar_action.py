"""Prepare a timestamp-normalized, aligned action without changing its package."""

import argparse
import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from avatars.wav2lip.face_detection.detection.sfd.sfd_detector import SFDDetector
from avatars.wav2lip.genavatar import generate_avatar
from server.avatar_video_validation import probe_video, validate_faces
from tools.align_avatar_video import align_video, box_stats


def edge_samples(frame):
    height, width = frame.shape[:2]
    band = max(2, round(width * .03))
    return np.concatenate((frame[:round(height * .08)].reshape(-1, 3),
                           frame[:, :band].reshape(-1, 3),
                           frame[:, -band:].reshape(-1, 3)))


def protected_background_shift(frame, correction, source_color):
    """Correct only gray regions connected to the canvas edge, protecting people."""
    distance = np.linalg.norm(frame.astype(np.float32) - source_color, axis=2)
    candidate = (distance < 45).astype(np.uint8)
    _, labels = cv2.connectedComponents(candidate)
    edge_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    background = np.isin(labels, edge_labels[edge_labels != 0]).astype(np.uint8)
    # Leave the silhouette unchanged and feather the correction into the background.
    weight = np.clip((cv2.distanceTransform(background, cv2.DIST_L2, 3) - 4) / 12, 0, 1)
    return np.clip(frame.astype(np.float32) + weight[..., None] * correction, 0, 255).astype(np.uint8)


def background_color(path, limit=25):
    capture = cv2.VideoCapture(str(path))
    samples = []
    try:
        for _ in range(limit):
            ok, frame = capture.read()
            if not ok:
                break
            samples.append(edge_samples(frame).mean(axis=0))
    finally:
        capture.release()
    if not samples:
        raise ValueError(f'No frames in {path}')
    return np.median(samples, axis=0)


def match_background(source, destination, reference):
    target_color = background_color(reference)
    source_color = background_color(source)
    correction = target_color - source_color + np.array([3, 2, 3])
    capture = cv2.VideoCapture(str(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    encoder = subprocess.Popen([
        'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', '25', '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(destination),
    ], stdin=subprocess.PIPE)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            encoder.stdin.write(protected_background_shift(frame, correction, source_color).tobytes())
    finally:
        capture.release()
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f'Background encoder failed: {code}')
    error = float(np.abs(background_color(destination) - target_color).mean())
    if error > 3:
        raise ValueError(f'Background edge error {error:.2f} exceeds 3 BGR levels')
    return {'reference_bgr': target_color.tolist(), 'edge_mae': error}


def prepare_action(input_path, reference, avatar_id, work, avatars, weights):
    work.mkdir(parents=True, exist_ok=True)
    destination = avatars / avatar_id
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    normalized = work / 'cfr25.mp4'
    aligned = work / 'aligned.mp4'
    source = work / 'prepared.mp4'
    # fps samples original timestamps, preserving duration for variable-rate clips.
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(input_path),
                    '-map', '0:v:0', '-vf', 'fps=25', '-an', '-c:v', 'libx264',
                    '-crf', '18', '-pix_fmt', 'yuv420p', str(normalized)], check=True)
    detector = SFDDetector('cuda', str(weights))
    reference_report = probe_video(reference, (786, 1178))
    reference_boxes = validate_faces(reference, reference_report, detector)
    target = box_stats(reference_boxes[:101])
    alignment = align_video(normalized, aligned, target, detector)
    background = match_background(aligned, source, reference)
    report = probe_video(source, (786, 1178))
    boxes = validate_faces(source, report, detector)
    measured = box_stats(boxes)
    if max(abs(measured[i] - target[i]) for i in range(2)) > 4 or abs(measured[2] - target[2]) > 4:
        raise ValueError(f'Face alignment is outside 4px tolerance: {measured} vs {target}')
    del detector
    avatars.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=avatars) as staging:
        generate_avatar(str(source), avatar_id, staging, img_size=256,
                        face_det_batch_size=4, validated_boxes=boxes)
        generated = Path(staging) / avatar_id
        with (generated / 'coords.pkl').open('rb') as handle:
            coords = pickle.load(handle)
        count = report['frame_count']
        if any(len(list((generated / kind).glob('*.png'))) != count for kind in ('full_imgs', 'face_imgs')) or len(coords) != count:
            raise ValueError('Incomplete generated avatar')
        os.replace(generated, destination)
    result = dict(avatar_id=avatar_id, validation=report, alignment=alignment,
                  measured_face=measured, background=background, source=str(source))
    (work / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--avatar-id', required=True)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--avatars', type=Path, default=Path('data/avatars'))
    parser.add_argument('--weights', type=Path, default=Path('models/s3fd.pth'))
    args = parser.parse_args()
    print(json.dumps(prepare_action(args.input, args.reference, args.avatar_id,
                                   args.work, args.avatars, args.weights), ensure_ascii=False))
