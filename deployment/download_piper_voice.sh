#!/usr/bin/env bash
set -euo pipefail

deployment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$deployment_dir")"
voice_dir="$project_dir/data/tts"
mkdir -p "$voice_dir"

curl_args=(-fL --retry 3)
if [[ "${CURL_INSECURE:-0}" == "1" ]]; then
    curl_args+=(-k)
fi

download_verified() {
    local name="$1" sha256="$2" url="$3"
    local target="$voice_dir/$name" temp="$voice_dir/$name.tmp.$$"
    if [[ -s "$target" ]] && echo "$sha256  $target" | sha256sum -c --status; then
        return
    fi
    rm -f "$temp"
    curl "${curl_args[@]}" "$url" -o "$temp"
    echo "$sha256  $temp" | sha256sum -c -
    mv "$temp" "$target"
}

base_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/zh/zh_CN/huayan/medium"
download_verified \
    zh_CN-huayan-medium.onnx \
    9929917bf8cabb26fd528ea44d3a6699c11e87317a14765312420be230be0f3d \
    "$base_url/zh_CN-huayan-medium.onnx?download=true"
download_verified \
    zh_CN-huayan-medium.onnx.json \
    d521dc45504a8ccc99e325822b35946dd701840bfb07e3dbb31a40929ed6a82b \
    "$base_url/zh_CN-huayan-medium.onnx.json?download=true"
