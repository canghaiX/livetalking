from threading import Thread
import queue
from queue import Queue
from io import BytesIO
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger
from .text_normalizer import prepare_speech_message

class State(Enum):
    RUNNING = 0
    PAUSE = 1

class BaseTTS:
    def __init__(self, opt, parent: "BaseAvatar"):
        self.opt = opt
        self.parent = parent

        #self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        with self.msgqueue.mutex:
            self.msgqueue.queue.clear()
            self.msgqueue.not_full.notify_all()
        self.state = State.PAUSE
        # Specialized streaming engines can clear generated-but-not-played
        # segments without making the base queue aware of their internals.
        clear_prefetch = getattr(self, '_clear_prefetch_queue', None)
        if clear_prefetch:
            clear_prefetch()

    def put_msg_txt(self, msg: str, datainfo: dict = None):
        if len(msg) > 0:
            self.msgqueue.put((msg, dict(datainfo or {})))

    def render(self, quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self, quit_event):        
        while not quit_event.is_set():
            try:
                msg: tuple[str, dict] = self.msgqueue.get(block=True, timeout=1)
                if msg[1].get('_speech_epoch', 0) != getattr(getattr(self, 'parent', None), 'speech_epoch', 0):
                    continue
                self.state = State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(prepare_speech_message(*msg))
        self.stop_tts()
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self, msg: tuple[str, dict]):
        pass

    def stop_tts(self):
        pass
