import asyncio
import threading
import time
import unittest

from server.webrtc import HumanPlayer
from server.session_manager import SessionManager


class Avatar:
    def __init__(self):
        self.done = threading.Event()

    def render(self, quit_event):
        quit_event.wait(2)
        time.sleep(.3)
        self.done.set()


class ShutdownTests(unittest.TestCase):
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


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_builder_removes_placeholder(self):
        manager = SessionManager()
        def fail(*args):
            raise ValueError('missing avatar')
        manager.init_builder(fail)
        with self.assertRaises(ValueError):
            await manager.create_session({},sessionid='failed')
        self.assertNotIn('failed',manager.sessions)


if __name__ == '__main__':
    unittest.main()
