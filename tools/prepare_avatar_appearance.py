"""Create a versioned Wav2Lip Avatar with canonical presenter lighting."""

import argparse
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import cv2

from avatars.wav2lip.face_detection.detection.sfd.sfd_detector import SFDDetector
from avatars.wav2lip.genavatar import generate_avatar
from server.avatar_video_validation import probe_video, validate_faces
from tools.normalize_avatar_appearance import (
    _read_sample_frames,
    appearance_profile,
    edge_color,
    normalize_video,
)


def prepare(input_path, reference_path, avatar_id, work, avatars, weights):
    work = Path(work)
    avatars = Path(avatars)
    destination = avatars / avatar_id
    if destination.exists():
        raise FileExistsError(f'拒绝覆盖已有 Avatar: {destination}')
    work.mkdir(parents=True, exist_ok=True)

    detector = SFDDetector('cuda', str(weights))
    reference_report = probe_video(reference_path, (786, 1178))
    reference_boxes = validate_faces(reference_path, reference_report, detector)
    source_report = probe_video(input_path, (786, 1178))
    source_boxes = validate_faces(input_path, source_report, detector)
    reference_frames = _read_sample_frames(reference_path, 101)
    target_profile = appearance_profile(reference_frames, reference_boxes[:len(reference_frames)])
    target_background = edge_color(reference_frames[0])
    normalized = work / 'appearance_fixed.mp4'
    appearance = normalize_video(input_path, normalized, target_profile, source_boxes,
                                 target_background=target_background)
    normalized_report = probe_video(normalized, (786, 1178))
    normalized_boxes = validate_faces(normalized, normalized_report, detector)
    del detector

    staging = Path(tempfile.mkdtemp(prefix='.appearance-', dir=str(avatars)))
    try:
        generate_avatar(str(normalized), avatar_id, str(staging), img_size=256,
                        pads=[0, 10, 0, 0], face_det_batch_size=4,
                        validated_boxes=normalized_boxes)
        generated = staging / avatar_id
        with (generated / 'coords.pkl').open('rb') as stream:
            coords = pickle.load(stream)
        frame_count = normalized_report['frame_count']
        if len(coords) != frame_count:
            raise ValueError('coords.pkl 帧数不完整')
        for kind in ('full_imgs', 'face_imgs'):
            if len(list((generated / kind).glob('*.png'))) != frame_count:
                raise ValueError(f'{kind} 帧数不完整')
        avatars.mkdir(parents=True, exist_ok=True)
        os.replace(generated, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    report = {'avatar_id': avatar_id, 'input': str(input_path),
              'reference': str(reference_path), 'validation': normalized_report,
              'appearance': appearance}
    (work / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--avatar-id', required=True)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--avatars', default='data/avatars', type=Path)
    parser.add_argument('--weights', default='models/s3fd.pth', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.reference, args.avatar_id, args.work,
                              args.avatars, args.weights), ensure_ascii=False))
