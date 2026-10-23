import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from server.webrtc import HumanPlayer
from server.session_manager import SessionManager
from avatars.base_avatar import AudioFrameData


class Avatar:
    def __init__(self):
        self.done = threading.Event()

    def render(self, quit_event):
        quit_event.wait(2)
        time.sleep(.3)
        self.done.set()


class ShutdownTests(unittest.TestCase):
    def test_interrupt_flushes_media_and_rejects_stale_generation(self):
        avatar = Avatar()
        avatar.speech_epoch = 0
        player = HumanPlayer(avatar)
        player.audio._queue.put((object(), {'_speech_epoch': 0}))
        player.video._queue.put((object(), {'_speech_epoch': 0}))
        avatar.speech_epoch = 1
        player.flush()
        self.assertTrue(player.audio._queue.empty())
        self.assertTrue(player.video._queue.empty())
        self.assertFalse(player.is_current({'_speech_epoch': 0}))
        self.assertTrue(player.is_current({'_speech_epoch': 1}))
        player.video.stop()
        player.audio.stop()

    def test_full_queue_unblocks_when_track_stops(self):
        player = HumanPlayer(Avatar())
        for _ in range(100):
            player.video._queue.put((None,None))
        sender = threading.Thread(target=player._push_frame,args=(player.video,object()))
        sender.start()
        player.video.stop()
        sender.join(1)
        self.assertFalse(sender.is_alive())
        player.audio.stop()

    def test_worker_join_does_not_block_stop(self):
        avatar = Avatar()
        player = HumanPlayer(avatar)
        player._start(player.video)
        start = time.monotonic()
        player.video.stop()
        self.assertLess(time.monotonic()-start,.2)
        self.assertTrue(avatar.done.wait(2))
        player.audio.stop()

    def test_audio_and_video_tracks_share_initial_clock(self):
        avatar = Avatar()
        avatar.opt = SimpleNamespace(webrtc_sync_buffer_frames=3)
        player = HumanPlayer(avatar)
        start = player.timeline_start()
        self.assertEqual(player.timeline_start(), start)
        player.video.stop()
        player.audio.stop()


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_builder_removes_placeholder(self):
        manager = SessionManager()
        def fail(*args):
            raise ValueError('missing avatar')
        manager.init_builder(fail)
        with self.assertRaises(ValueError):
            await manager.create_session({},sessionid='failed')
        self.assertNotIn('failed',manager.sessions)


class TimelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.avatar = Avatar()
        self.avatar.speech_epoch = 0
        self.player = HumanPlayer(self.avatar)

    def tearDown(self):
        self.player.audio.stop()
        self.player.video.stop()

    async def test_both_tracks_need_three_video_frames_of_preroll(self):
        player = self.player
        player._HumanPlayer__started.update([player.audio, player.video])
        wait = asyncio.create_task(player.wait_for_sync_buffer())
        for _ in range(6):
            player.audio._queue.put((object(), {}))
        for _ in range(2):
            player.video._queue.put((object(), {}))
        await asyncio.sleep(.02)
        self.assertFalse(wait.done())
        player.video._queue.put((object(), {}))
        await asyncio.wait_for(wait, .2)
        self.assertTrue(player._sync_ready)
        start = player._sync_start
        await player.wait_for_sync_buffer()
        self.assertEqual(start, player._sync_start)

    async def test_explicit_sample_pts_stay_aligned_for_ten_minutes(self):
        player = self.player
        player._sync_start = 100.
        with patch('server.webrtc.time.monotonic', return_value=10000.):
            for i in range(0, 15001, 17):
                sample = i * 640
                video_pts, video_base = await player.video.next_timestamp(sample)
                audio_pts, audio_base = await player.audio.next_timestamp(sample)
                next_audio_pts, _ = await player.audio.next_timestamp(sample + 320)
                self.assertEqual(video_pts * video_base, audio_pts * audio_base)
                self.assertEqual(next_audio_pts - audio_pts, 320)
                self.assertEqual(player.video._start, player.audio._start)

    async def test_media_pair_and_interrupt_share_the_same_position(self):
        player = self.player
        frames = [AudioFrameData(np.ones(320, dtype=np.float32) * .1, 0,
                                 {'_speech_epoch': 0}) for _ in range(2)]
        video = np.zeros((16, 16, 3), dtype=np.uint8)
        player.push_media(video, frames, 0)
        audio = [player.audio._queue.get_nowait()[1] for _ in range(2)]
        image = player.video._queue.get_nowait()[1]
        self.assertEqual([a['_media_sample_start'] for a in audio], [0, 320])
        self.assertEqual(image['_media_sample_start'], 0)
        player._sent_sample_end.update(audio=640, video=640)
        self.avatar.speech_epoch = 1
        player.flush()
        player.push_media(video, frames, 0)
        self.assertTrue(player.video._queue.empty())
        player.push_media(video, frames, 1)
        image = player.video._queue.get_nowait()[1]
        audio = player.audio._queue.get_nowait()[1]
        self.assertEqual(image['_media_sample_start'], 640)
        self.assertEqual(image['_media_sample_start'], audio['_media_sample_start'])

    async def test_video_can_be_presented_at_mel_window_center(self):
        player = self.player
        frames = [AudioFrameData(np.ones(320, dtype=np.float32) * .1, 0,
                                 {'_speech_epoch': 0}) for _ in range(2)]
        player.push_media(np.zeros((16, 16, 3), dtype=np.uint8), frames, 0, 100)
        audio_event = player.audio._queue.get_nowait()[1]
        video_event = player.video._queue.get_nowait()[1]
        self.assertEqual(audio_event['_media_sample_start'], 0)
        self.assertEqual(video_event['_media_sample_start'], 1600)


if __name__ == '__main__':
    unittest.main()
