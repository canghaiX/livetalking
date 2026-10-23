"""Report the production lip-sync timing sweep used for manual A/B review."""

import argparse
import json
import wave
from pathlib import Path

import numpy as np


def offsets(limit=120, step=20):
    return list(range(-limit, limit + 1, step))


def audio_onset_ms(audio_path, threshold_db=-35.0):
    """Find the first energetic 10 ms PCM window in a mono WAV file."""
    with wave.open(str(audio_path), 'rb') as source:
        rate = source.getframerate()
        channels = source.getnchannels()
        width = source.getsampwidth()
        if width != 2:
            raise ValueError('校准音频必须是16-bit PCM WAV')
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype='<i2')
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if not len(samples):
        raise ValueError('校准音频为空')
    block = max(1, round(rate * .01))
    rms = np.array([np.sqrt(np.mean(np.square(samples[i:i + block], dtype=np.float64)))
                    for i in range(0, len(samples), block)])
    peak = max(float(rms.max()), 1.0)
    threshold = peak * (10.0 ** (threshold_db / 20.0))
    hits = np.flatnonzero(rms >= threshold)
    return float(hits[0] * block * 1000.0 / rate) if len(hits) else 0.0


def select_offset(audio_start_ms, mouth_start_ms, limit=120, step=20):
    """Select the nearest tested offset from measured audio/mouth onsets."""
    measured = float(mouth_start_ms) - float(audio_start_ms)
    value = min(offsets(limit, step), key=lambda candidate: abs(candidate - measured))
    return value, measured


def calibration_plan(audio_path, limit=120, step=20, mouth_start_ms=None):
    result = {
        'audio': str(audio_path),
        'offsets_ms': offsets(limit, step),
        'max_extra_startup_ms': 300,
        'instructions': [
            '使用同一份Serena音频依次播放每个偏移版本',
            '记录第一个明显张嘴帧与音频能量起点的时间差',
            '选择持续偏差最小且不超过一帧40ms的偏移',
            '若所有偏移都无法改善中文音素形状，记录为模型拟合限制而不是继续增加延迟',
        ],
    }
    if mouth_start_ms is not None:
        audio_start = audio_onset_ms(audio_path)
        chosen, measured = select_offset(audio_start, mouth_start_ms, limit, step)
        result.update(audio_onset_ms=audio_start, measured_offset_ms=measured,
                      selected_offset_ms=chosen, selection='measured')
    else:
        result['selection'] = 'manual_review_required'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=120)
    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--mouth-onset-ms', type=float,
                        help='从视频帧标记得到的首个明显张嘴时间；提供后自动选择最近偏移')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = calibration_plan(args.audio, args.limit, args.step, args.mouth_onset_ms)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
