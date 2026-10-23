"""Check and describe a reproducible MuseTalk/Wav2Lip A/B run.

The production service intentionally remains on Wav2Lip.  MuseTalk requires a
separate set of large weights, so this command fails clearly until those files
are installed in the persistent ``models`` volume instead of downloading them
implicitly during a live session.
"""

import argparse
import json
from pathlib import Path


REQUIRED = {
    "models/musetalkV15/unet.pth": "MuseTalk v1.5 UNet",
    "models/musetalkV15/musetalk.json": "MuseTalk UNet config",
    "models/sd-vae": "Stable Diffusion VAE directory",
    "models/whisper": "Whisper feature model directory",
}


def check_models(root: Path):
    missing = []
    present = []
    for relative, label in REQUIRED.items():
        path = root / relative
        exists = path.is_dir() if relative.endswith("vae") or relative.endswith("whisper") else path.is_file()
        (present if exists else missing).append({"path": relative, "label": label})
    return {"ready": not missing, "present": present, "missing": missing}


def build_report(root: Path, audio: Path | None = None):
    result = check_models(root)
    result.update({
        "production_model": "wav2lip",
        "candidate_model": "musetalk-v1.5",
        "audio": str(audio) if audio else None,
        "comparison": [
            "中文声母、韵母嘴形和嘴部起止",
            "眼睛与脸部稳定性",
            "GPU显存、首帧等待和持续播放延迟",
        ],
    })
    if not result["ready"]:
        result["status"] = "blocked_missing_weights"
        result["message"] = "MuseTalk权重未齐全；继续使用生产Wav2Lip，避免运行时隐式下载。"
    else:
        result["status"] = "ready_for_ab_test"
        result["message"] = "权重齐全，可用同一Serena音频执行独立A/B测试；不会自动切换生产模型。"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.models, args.audio)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["ready"] else 2)
