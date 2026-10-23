"""Build a versioned Wav2Lip Avatar after background plate composition."""

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path

from avatars.wav2lip.face_detection.detection.sfd.sfd_detector import SFDDetector
from avatars.wav2lip.genavatar import generate_avatar
from server.avatar_video_validation import probe_video, validate_faces
from tools.align_avatar_video import box_stats
from tools.compose_avatar_background import compose


def prepare(input_path, reference, avatar_id, work, avatars, weights):
    work.mkdir(parents=True, exist_ok=True)
    destination = avatars / avatar_id
    if destination.exists():
        raise FileExistsError(f'拒绝覆盖已有Avatar: {destination}')
    composed = work / 'prepared.mp4'
    background = compose(input_path, composed, reference)
    detector = SFDDetector('cuda', str(weights))
    reference_report = probe_video(reference, (786, 1178))
    reference_boxes = validate_faces(reference, reference_report, detector)
    target = box_stats(reference_boxes[:101])
    report = probe_video(composed, (786, 1178))
    boxes = validate_faces(composed, report, detector)
    measured = box_stats(boxes)
    if max(abs(measured[i] - target[i]) for i in range(2)) > 4 or abs(measured[2] - target[2]) > 4:
        raise ValueError(f'人物构图超出4像素容差: {measured} vs {target}')
    del detector
    with tempfile.TemporaryDirectory(prefix='.prepare-bgfix-', dir=avatars) as staging:
        generate_avatar(str(composed), avatar_id, staging, img_size=256,
                        face_det_batch_size=4, validated_boxes=boxes)
        generated = Path(staging) / avatar_id
        with (generated / 'coords.pkl').open('rb') as handle:
            coords = pickle.load(handle)
        count = report['frame_count']
        if len(coords) != count or any(len(list((generated / kind).glob('*.png'))) != count
                                       for kind in ('full_imgs', 'face_imgs')):
            raise ValueError('Avatar输出帧不完整')
        os.replace(generated, destination)
    result = {'avatar_id': avatar_id, 'validation': report, 'background': background,
              'target_face': target, 'measured_face': measured, 'source': str(composed)}
    (work / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--avatar-id', required=True)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--avatars', default='data/avatars', type=Path)
    parser.add_argument('--weights', default='models/s3fd.pth', type=Path)
    args = parser.parse_args()
    args.avatars.mkdir(parents=True, exist_ok=True)
    print(json.dumps(prepare(args.input, args.reference, args.avatar_id, args.work,
                              args.avatars, args.weights), ensure_ascii=False))
