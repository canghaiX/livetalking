#!/usr/bin/env bash
set -euo pipefail
deployment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$deployment_dir")"
if [[ ! -s "$project_dir/data/tts/zh_CN-huayan-medium.onnx" || \
      ! -s "$project_dir/data/tts/zh_CN-huayan-medium.onnx.json" ]]; then
    "$deployment_dir/download_piper_voice.sh"
fi
if docker container inspect livetalking >/dev/null 2>&1; then
    docker start livetalking
    exit 0
fi
mkdir -p "$project_dir/data/record" "$project_dir/data/avatar_packages" "$project_dir/data/avatars" "$project_dir/data/tts"
# Use the local HTTP proxy exposed on the server. This is the proxy path that
# is currently authorized for outbound DeepSeek access on this network.
docker run -d \
    --name livetalking \
    --gpus device=0 \
    --network host \
    --restart unless-stopped \
    --init \
    --shm-size 1g \
    --env-file "$deployment_dir/secrets.env" \
    -e HTTP_PROXY=http://127.0.0.1:17897 \
    -e HTTPS_PROXY=http://127.0.0.1:17897 \
    -e http_proxy=http://127.0.0.1:17897 \
    -e https_proxy=http://127.0.0.1:17897 \
    -e ALL_PROXY= \
    -e all_proxy= \
    -e NO_PROXY=localhost,127.0.0.1,::1 \
    --log-opt max-size=20m --log-opt max-file=3 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e LIVETALKING_ALLOW_RESTART=1 \
    -e LIVETALKING_SFD_PATH=/opt/livetalking/models/s3fd.pth \
    -v "$deployment_dir/config.yaml:/opt/livetalking/deployment/config.yaml:ro" \
    -v "$deployment_dir/avatar_motion.json:/opt/livetalking/deployment/avatar_motion.json:ro" \
    -v "$project_dir/data/record:/opt/livetalking/data/record" \
    -v "$project_dir/data/avatar_packages:/opt/livetalking/data/avatar_packages" \
    -v "$project_dir/data/avatars:/opt/livetalking/data/avatars" \
    -v "$project_dir/data/tts:/opt/livetalking/data/tts" \
    livetalking:latest
