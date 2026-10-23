import base64
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tts.base_tts import State
from tts.qwentts import QwenTTS


class Sink:
    def __init__(self):
        self.frames = []

    def put_audio_frame(self, frame, event):
        self.frames.append((frame, dict(event)))


class FakeQwen:
    instances = []
    commits_before_audio = 0

    def __init__(self, model, callback, url):
        self.callback = callback
        self.appended = []
        self.closed = False
        FakeQwen.instances.append(self)

    def connect(self):
        self.callback.on_open()

    def update_session(self, **kwargs):
        self.session = kwargs

    def append_text(self, text):
        self.appended.append(text)

    def commit(self):
        if FakeQwen.commits_before_audio:
            FakeQwen.commits_before_audio -= 1
            return
        pcm = (np.ones(480, dtype=np.int16) * 700).tobytes()
        self.callback.on_event({'type': 'response.audio.delta',
                                'delta': base64.b64encode(pcm).decode()})
        self.callback.on_event({'type': 'response.done'})

    def close(self):
        self.closed = True


def make_tts():
    FakeQwen.instances = []
    FakeQwen.commits_before_audio = 0
    parent = Sink()
    opt = SimpleNamespace(fps=25, REF_FILE='Cherry',
                          qwen_tts_retry_initial_seconds=0.2,
                          qwen_tts_retry_max_seconds=0.2,
                          qwen_tts_response_timeout_seconds=1)
    with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test-key'}):
        tts = QwenTTS(opt, parent)
    tts.state = State.RUNNING
    return tts, parent


class QwenTtsTests(unittest.TestCase):
    def test_complete_segment_uses_serena_and_emits_end_after_response_done(self):
        tts, parent = make_tts()
        with patch('tts.qwentts.QwenTtsRealtime', FakeQwen):
            tts.txt_to_audio(('最后一句也必须播报。', {'_speech_epoch': 0}))
        self.assertEqual(tts.voice, 'Serena')
        self.assertEqual(FakeQwen.instances[0].session['voice'], 'Serena')
        self.assertEqual(FakeQwen.instances[0].appended, ['最后一句也必须播报。'])
        self.assertEqual(parent.frames[0][1]['status'], 'start')
        self.assertEqual(parent.frames[-1][1]['status'], 'end')
        self.assertTrue(np.any(parent.frames[0][0]))

    def test_empty_response_retries_same_text_without_fallback(self):
        tts, parent = make_tts()
        FakeQwen.commits_before_audio = 1
        with patch('tts.qwentts.QwenTtsRealtime', FakeQwen):
            with patch('tts.edge.EdgeTTS') as edge:
                tts.txt_to_audio(('尾部重试文本。', {}))
        self.assertEqual(len(FakeQwen.instances), 2)
        self.assertEqual([item.appended for item in FakeQwen.instances],
                         [['尾部重试文本。'], ['尾部重试文本。']])
        edge.assert_not_called()
        self.assertEqual(parent.frames[-1][1]['status'], 'end')

    def test_successive_segments_reuse_same_serena_session(self):
        tts, parent = make_tts()
        with patch('tts.qwentts.QwenTtsRealtime', FakeQwen):
            tts.txt_to_audio(('第一段。', {}))
            tts.txt_to_audio(('第二段。', {}))
        self.assertEqual(len(FakeQwen.instances), 1)
        self.assertEqual(FakeQwen.instances[0].appended, ['第一段。', '第二段。'])
        self.assertEqual(sum(event.get('status') == 'start' for _, event in parent.frames), 2)
        self.assertEqual(sum(event.get('status') == 'end' for _, event in parent.frames), 2)

    def test_flush_interrupts_retry_and_old_text_is_not_emitted(self):
        tts, parent = make_tts()
        tts.retry_initial_seconds = tts.retry_max_seconds = 0.5
        FakeQwen.commits_before_audio = 100
        with patch('tts.qwentts.QwenTtsRealtime', FakeQwen):
            worker = threading.Thread(target=tts.txt_to_audio, args=(('被打断的尾句。', {}),))
            worker.start()
            time.sleep(0.25)
            tts.flush_talk()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(parent.frames, [])


if __name__ == '__main__':
    unittest.main()
