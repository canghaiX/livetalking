"""Measure decoded flash/tone alignment through the production WebRTC player.

Run inside the application image with this script on stdin. No TTS or model
API calls are needed; markers exercise the encoder, RTP, and receiver timeline.
"""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection

from avatars.base_avatar import AudioFrameData
from server.webrtc import HumanPlayer


class MarkerAvatar:
    speech_epoch = 0
    opt = SimpleNamespace(webrtc_sync_buffer_frames=3)

    def notify(self, event):
        pass

    def render(self, quit_event):
        index = 0
        while not quit_event.is_set():
            marker = index >= 50 and index % 50 == 0
            frame = np.full((128, 128, 3), 255 if marker else 0, dtype=np.uint8)
            samples = np.zeros(640, dtype=np.float32)
            if marker:
                samples = (.35 * np.sin(np.arange(640) * 2 * np.pi * 1000 / 16000)).astype(np.float32)
            pair = [AudioFrameData(samples[:320], 0), AudioFrameData(samples[320:], 0)]
            self.player.push_media(frame, pair, 0)
            index += 1


async def main():
    avatar = MarkerAvatar()
    avatar.player = HumanPlayer(avatar)
    player = avatar.player
    config = RTCConfiguration(iceServers=[])
    sender, receiver = RTCPeerConnection(config), RTCPeerConnection(config)
    tasks = []
    markers = {'audio': [], 'video': []}

    async def consume(track):
        while True:
            frame = await track.recv()
            seconds = float(frame.pts * frame.time_base)
            if track.kind == 'video':
                if frame.to_ndarray(format='bgr24').mean() > 200:
                    markers['video'].append(seconds)
            else:
                channels = len(frame.layout.channels)
                values = frame.to_ndarray().reshape(-1, channels).astype(np.int32)
                active = np.flatnonzero(np.abs(values).max(axis=1) > 1000)
                if active.size:
                    onset = seconds + int(active[0]) / frame.sample_rate
                    if not markers['audio'] or onset - markers['audio'][-1] > 1:
                        markers['audio'].append(onset)

    @receiver.on('track')
    def on_track(track):
        tasks.append(asyncio.create_task(consume(track)))

    try:
        receiver.addTransceiver('audio', direction='recvonly')
        receiver.addTransceiver('video', direction='recvonly')
        sender.addTrack(player.audio)
        sender.addTrack(player.video)
        await receiver.setLocalDescription(await receiver.createOffer())
        await sender.setRemoteDescription(receiver.localDescription)
        await sender.setLocalDescription(await sender.createAnswer())
        await receiver.setRemoteDescription(sender.localDescription)
        for _ in range(250):
            if min(map(len, markers.values())) >= 5:
                break
            await asyncio.sleep(.1)
        count = min(map(len, markers.values()))
        assert count >= 5, markers
        offsets = [round((markers['video'][i] - markers['audio'][i]) * 1000, 3)
                   for i in range(count)]
        result = dict(markers=markers, video_minus_audio_ms=offsets,
                      max_abs_offset_ms=max(map(abs, offsets)))
        print(json.dumps(result))
        assert result['max_abs_offset_ms'] <= 40, result
    finally:
        await receiver.close()
        await sender.close()
        player.audio.stop()
        player.video.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


asyncio.run(asyncio.wait_for(main(), 40))
