import unittest

import numpy as np

from tts.edge import EdgeTTS


class AudioSink:
    def __init__(self):
        self.frames = []

    def put_audio_frame(self, frame, event):
        self.frames.append((frame, event))


class LocalTtsFallbackTests(unittest.TestCase):
    def test_local_mandarin_voice_produces_audio_frames(self):
        tts = EdgeTTS.__new__(EdgeTTS)
        tts.sample_rate = 16000
        tts.chunk = 320
        tts.parent = AudioSink()

        tts._local_tts("欢迎参观铁路科研展厅。", {"avatar_action": "smile"})

        self.assertGreater(len(tts.parent.frames), 1)
        self.assertEqual(len(tts.parent.frames[0][0]), 320)
        self.assertEqual(tts.parent.frames[0][1]["status"], "start")
        self.assertEqual(tts.parent.frames[-1][1]["status"], "end")
        self.assertTrue(any(np.any(frame) for frame, _ in tts.parent.frames))


if __name__ == "__main__":
    unittest.main()
