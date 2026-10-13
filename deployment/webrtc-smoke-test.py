"""Verify local WebRTC audio/video and audio-driven inference after deployment."""
import asyncio
import io
import json
import wave
import os

import aiohttp
import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription


async def main():
    base = os.environ.get('SMOKE_BASE_URL', 'http://127.0.0.1:8020')
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    tasks = []
    frames = {"audio": 0, "video": 0}
    peak = 0

    async def consume(track):
        nonlocal peak
        while True:
            frame = await track.recv()
            frames[track.kind] += 1
            if track.kind == "audio":
                peak = max(peak, int(np.abs(frame.to_ndarray().astype(np.int32)).max()))

    @pc.on("track")
    def on_track(track):
        tasks.append(asyncio.create_task(consume(track)))

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40)) as http:
            pc.addTransceiver("audio", direction="recvonly")
            pc.addTransceiver("video", direction="recvonly")
            await pc.setLocalDescription(await pc.createOffer())
            async with http.post(base + "/offer", json={
                "sdp": pc.localDescription.sdp, "type": pc.localDescription.type,
                **({'avatar': os.environ['SMOKE_AVATAR']} if os.environ.get('SMOKE_AVATAR') else {})
            }) as response:
                response.raise_for_status()
                answer = await response.json()
            await pc.setRemoteDescription(RTCSessionDescription(answer["sdp"], answer["type"]))
            for _ in range(300):
                if frames["video"] >= 10 and frames["audio"] >= 10:
                    break
                await asyncio.sleep(0.1)
            assert frames["video"] >= 10 and frames["audio"] >= 10, frames

            # A local waveform exercises lip inference without an external TTS dependency.
            samples = (np.sin(np.arange(32000) * 2 * np.pi * 220 / 16000) * 10000).astype("<i2")
            wav = io.BytesIO()
            with wave.open(wav, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(samples.tobytes())
            form = aiohttp.FormData()
            form.add_field("sessionid", str(answer["sessionid"]))
            form.add_field("file", wav.getvalue(), filename="test.wav", content_type="audio/wav")
            async with http.post(base + "/humanaudio", data=form) as response:
                result = await response.json()
                assert result["code"] == 0, result
            for _ in range(150):
                if peak > 500:
                    break
                await asyncio.sleep(0.1)
            assert peak > 500, "No non-silent audio received"
            assert pc.connectionState == "connected", pc.connectionState
            print(json.dumps({"connection": pc.connectionState, "received_frames": frames,
                              "audio_peak": peak, "audio_upload": result}, ensure_ascii=False))
    finally:
        await pc.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


asyncio.run(asyncio.wait_for(main(), timeout=65))
