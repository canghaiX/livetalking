"""Alibaba Qwen realtime TTS with a fixed Serena voice."""

import base64
import os
import queue
import threading
import time
import uuid

import numpy as np

from registry import register
from utils.logger import logger
from .base_tts import BaseTTS, State
from .pcm_resampler import PcmResampler
from .text_normalizer import prepare_speech_message

try:
    import dashscope
    from dashscope.audio.qwen_tts_realtime import (
        AudioFormat,
        QwenTtsRealtime,
        QwenTtsRealtimeCallback,
    )
except ImportError:
    logger.error("QwenTTS 需要安装 dashscope SDK: pip install dashscope>=1.25.11")
    raise


SRC_SR = 24000
DST_SR = 16000


@register("tts", "qwentts")
class QwenTTS(BaseTTS):
    """Qwen3-TTS realtime client. No EdgeTTS/Piper fallback is allowed."""

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.voice = "Serena"
        self.model = getattr(opt, "qwen_tts_model", "qwen3-tts-flash-realtime")
        self.ws_url = getattr(
            opt, "qwen_tts_url", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
        )
        self.retry_initial_seconds = max(
            0.2, float(getattr(opt, "qwen_tts_retry_initial_seconds", 1.0))
        )
        self.retry_max_seconds = max(
            self.retry_initial_seconds,
            float(getattr(opt, "qwen_tts_retry_max_seconds", 8.0)),
        )
        self.response_timeout_seconds = max(
            5.0, float(getattr(opt, "qwen_tts_response_timeout_seconds", 60.0))
        )
        self.first_buffer_frames = max(
            1, int(getattr(opt, "qwen_tts_first_buffer_frames", 3))
        )
        api_key = getattr(opt, "dashscope_api_key", None) or os.environ.get(
            "DASHSCOPE_API_KEY"
        )
        if not api_key:
            raise RuntimeError(
                "QwenTTS requires DASHSCOPE_API_KEY; no voice fallback is enabled"
            )
        dashscope.api_key = api_key

        self._client = None
        self._client_lock = threading.RLock()
        self._response_event = threading.Event()
        self._cancel_event = threading.Event()
        self._attempt_token = ""
        self._client_token = ""
        self._attempt_done = False
        self._attempt_error = ""
        self._attempt_frames = []
        self._attempt_remainder = np.array([], dtype=np.float32)
        self._attempt_started = False
        self._attempt_had_audio = False
        self._attempt_end_sent = False
        self._audio_lock = threading.RLock()
        self._resampler = PcmResampler(SRC_SR, DST_SR)
        self._current_text = ""
        self._current_event = {}
        self._audio_frames_sent = 0
        logger.info("QwenTTS 初始化完成: model=%s, voice=%s", self.model, self.voice)

        # Realtime generation and media delivery are intentionally decoupled.
        # A short bounded queue masks a transient provider/network stall while
        # keeping memory bounded and preserving segment order.
        self.prefetch_queue = queue.Queue(maxsize=max(1, int(
            getattr(opt, "qwen_tts_prefetch_queue_size", 4))))
        self.prefetch_min_segments = max(
            1, int(getattr(opt, "qwen_tts_prefetch_segments", 1))
        )
        self._prefetch_quit = None
        self._producer_thread = None
        self._player_thread = None
        self._collect_mode = False
        self._segment_sequence = 0

    def render(self, quit_event):
        """Run a producer and a playback consumer for streamed TTS segments."""
        self._prefetch_quit = quit_event
        self._producer_thread = threading.Thread(
            target=self._prefetch_loop, args=(quit_event,), name="qwen-tts-producer",
            daemon=True,
        )
        self._player_thread = threading.Thread(
            target=self._playback_loop, args=(quit_event,), name="qwen-tts-player",
            daemon=True,
        )
        self._producer_thread.start()
        self._player_thread.start()

    def _clear_prefetch_queue(self):
        with self.prefetch_queue.mutex:
            self.prefetch_queue.queue.clear()
            self.prefetch_queue.not_full.notify_all()

    def _prefetch_loop(self, quit_event):
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(timeout=.2)
            except queue.Empty:
                continue
            event = msg[1] if len(msg) > 1 else {}
            if event.get('_speech_epoch', 0) != getattr(
                    getattr(self, 'parent', None), 'speech_epoch', 0):
                continue
            self.state = State.RUNNING
            segment = self._synthesize_segment(prepare_speech_message(*msg))
            if segment is None:
                continue
            while not quit_event.is_set() and not self._cancel_event.is_set():
                try:
                    self.prefetch_queue.put(segment, timeout=.1)
                    break
                except queue.Full:
                    continue
        self._cancel_event.set()
        self._response_event.set()

    def _playback_loop(self, quit_event):
        first_ready = False
        while not quit_event.is_set():
            try:
                segment = self.prefetch_queue.get(timeout=.2)
            except queue.Empty:
                continue
            # Wait for one complete successor before starting a non-final
            # segment. This is the network jitter buffer; a one-segment answer
            # starts immediately because it is marked final.
            current_epoch = segment['event'].get('_speech_epoch', 0)
            if not first_ready and not segment['final']:
                # Do not wait forever when the LLM is still producing the
                # next segment.  The bounded deadline preserves the jitter
                # buffer while allowing a single segment to start promptly.
                deadline = time.monotonic() + 0.8
                while (self.prefetch_queue.qsize() < self.prefetch_min_segments
                       and not quit_event.is_set()
                       and not self._cancel_event.is_set()):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(.01)
            first_ready = True
            if self._cancel_event.is_set():
                continue
            event = dict(segment['event'])
            frames = segment['frames']
            for index, frame in enumerate(frames):
                if quit_event.is_set() or self._cancel_event.is_set():
                    break
                if index == 0:
                    start_event = dict(event, status='start', text=segment.get('text', event.get('text', '')))
                else:
                    start_event = event
                self.parent.put_audio_frame(frame, start_event)
            else:
                end_event = dict(event, status='end', text=segment.get('text', event.get('text', '')),
                                 _audio_type=1)
                self.parent.put_audio_frame(
                    np.zeros(self.chunk, dtype=np.float32), end_event
                )
                logger.info(
                    "QwenTTS 播放完成: segment=%s frames=%d buffered=%d",
                    segment['sequence'], len(frames), self.prefetch_queue.qsize()
                )
        self.stop_tts()

    def _callback(self, token):
        owner = self

        class Callback(QwenTtsRealtimeCallback):
            def on_open(self):
                logger.info("QwenTTS WebSocket 连接已建立 (voice=%s)", owner.voice)

            def on_close(self, close_status_code, close_msg):
                if token == owner._client_token:
                    owner._attempt_error = f"connection closed: {close_status_code}"
                    owner._response_event.set()
                logger.info("QwenTTS WebSocket 关闭: code=%s, msg=%s", close_status_code, close_msg)

            def on_event(self, response):
                try:
                    if token != owner._client_token:
                        return
                    event_type = response.get("type", "")
                    if event_type == "session.created":
                        logger.info("QwenTTS session: %s", response.get("session", {}).get("id", ""))
                    elif event_type == "response.audio.delta":
                        value = response.get("delta", "")
                        if value:
                            owner._on_audio_data(base64.b64decode(value), token)
                    elif event_type == "response.done":
                        owner._attempt_done = True
                        owner._response_event.set()
                        logger.info("QwenTTS response done (text=%s)", owner._current_text[:40])
                    elif event_type == "error":
                        owner._attempt_error = "provider error"
                        owner._response_event.set()
                        logger.error("QwenTTS provider error: %s", response)
                except Exception:
                    owner._attempt_error = "callback error"
                    owner._response_event.set()
                    logger.exception("QwenTTS 回调处理异常")

        return Callback()

    def _close_client(self):
        with self._client_lock:
            client, self._client = self._client, None
            self._client_token = ""
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("QwenTTS close failed", exc_info=True)

    def _connect_client(self, token):
        client = QwenTtsRealtime(
            model=self.model,
            callback=self._callback(token),
            url=self.ws_url,
        )
        client.connect()
        client.update_session(
            voice=self.voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            sample_rate=SRC_SR,
            mode="commit",
        )
        with self._client_lock:
            if token != self._attempt_token or self._cancel_event.is_set():
                client.close()
                raise RuntimeError("QwenTTS attempt cancelled")
            self._client = client
            self._client_token = token

    def _reset_attempt(self, token):
        with self._audio_lock:
            self._attempt_token = token
            self._response_event.clear()
            self._attempt_done = False
            self._attempt_error = ""
            self._attempt_frames = []
            self._attempt_remainder = np.array([], dtype=np.float32)
            self._attempt_started = False
            self._attempt_had_audio = False
            self._attempt_end_sent = False
            self._resampler.reset()

    def txt_to_audio_legacy(self, msg: tuple[str, dict]):
        text, textevent = msg
        current_epoch = getattr(getattr(self, 'parent', None), 'speech_epoch',
                                textevent.get('_speech_epoch', 0))
        if textevent.get('_speech_epoch', current_epoch) != current_epoch:
            return None
        self._cancel_event.clear()
        self._current_text = text
        self._current_event = dict(textevent or {})
        self._audio_frames_sent = 0
        backoff = self.retry_initial_seconds
        attempt_number = 0

        while self.state == State.RUNNING and not self._cancel_event.is_set():
            if textevent.get('_speech_epoch', current_epoch) != getattr(
                    getattr(self, 'parent', None), 'speech_epoch', current_epoch):
                return None
            attempt_number += 1
            token = uuid.uuid4().hex
            self._reset_attempt(token)
            try:
                with self._client_lock:
                    client = self._client
                if client is None:
                    self._connect_client(token)
                    with self._client_lock:
                        client = self._client
                else:
                    logger.debug("复用 QwenTTS Serena 会话处理下一段文本")
                client.append_text(text)
                client.commit()
                completed = self._response_event.wait(self.response_timeout_seconds)
                with self._audio_lock:
                    attempt_done = self._attempt_done
                    attempt_started = self._attempt_started
                    has_pending_audio = bool(self._attempt_frames) or bool(
                        self._attempt_remainder.size)
                if completed and attempt_done and (attempt_started or has_pending_audio):
                    if self._collect_mode:
                        frames = self._finish_attempt(collect=True)
                        if frames:
                            self._segment_sequence += 1
                            logger.info(
                                "QwenTTS 合成完成: segment=%s attempt=%d frames=%d text=%s",
                                self._segment_sequence, attempt_number, len(frames), text[:40]
                            )
                            return {
                                'frames': frames,
                                'event': dict(self._current_event),
                                'text': self._current_event.get('_tts_display_text', self._current_text),
                                'sequence': self._segment_sequence,
                                'final': bool(self._current_event.get('_round_final', False)),
                            }
                    self._finish_attempt()
                    with self._audio_lock:
                        attempt_started = self._attempt_started
                success = (completed and attempt_done and attempt_started
                           and self.state == State.RUNNING
                           and not self._cancel_event.is_set())
                if success:
                    self._finish_attempt()
                    # Keep the same Serena session for the next queued
                    # segment.  This reduces speaker/prosody drift between
                    # short streaming sentences.
                    logger.info(
                        "QwenTTS 合成完成: attempt=%d, frames=%d, text=%s",
                        attempt_number, self._audio_frames_sent, text[:40]
                    )
                    return
                reason = self._attempt_error or ("timeout" if not completed else "empty audio")
                # Once a first frame has reached the media queue, replaying the
                # whole sentence would duplicate speech. Close this partial
                # utterance and let BaseTTS continue with the next one.
                if attempt_started:
                    logger.error("QwenTTS Serena interrupted after audio started: %s", reason)
                    self._finish_attempt(error=reason)
                    return
            except Exception as exc:
                reason = str(exc) or exc.__class__.__name__
            finally:
                with self._audio_lock:
                    keep_connection = (self._attempt_had_audio and self._attempt_done
                                       and not self._cancel_event.is_set())
                if not keep_connection:
                    self._close_client()

            if self._cancel_event.is_set() or self.state != State.RUNNING:
                return
            logger.warning(
                "QwenTTS Serena attempt %d failed (%s); retrying in %.1fs",
                attempt_number, reason, backoff
            )
            self._cancel_event.wait(backoff)
            backoff = min(self.retry_max_seconds, backoff * 2)

    def txt_to_audio(self, msg: tuple[str, dict]):
        """Compatibility API; normal rendering uses the prefetch workers."""
        return self.txt_to_audio_legacy(msg)

    def _synthesize_segment(self, msg: tuple[str, dict]):
        """Generate a complete segment before exposing it to media output."""
        self._collect_mode = True
        try:
            return self.txt_to_audio_legacy(msg)
        finally:
            self._collect_mode = False

    def _on_audio_data(self, pcm_data: bytes, token):
        with self._audio_lock:
            if (token != self._client_token or self._cancel_event.is_set()
                    or self.state != State.RUNNING):
                return
            samples = self._resampler.process(pcm_data)
            self._attempt_had_audio = True
            if self._attempt_remainder.size:
                samples = np.concatenate([self._attempt_remainder, samples])
            index = 0
            while len(samples) - index >= self.chunk:
                self._attempt_frames.append(samples[index:index + self.chunk].copy())
                index += self.chunk
            self._attempt_remainder = samples[index:]
            if self._collect_mode:
                return
            should_flush = (not self._attempt_started
                            and len(self._attempt_frames) >= self.first_buffer_frames)
            if self._attempt_started:
                frames = self._attempt_frames
                self._attempt_frames = []
            elif should_flush:
                frames = self._attempt_frames
                self._attempt_frames = []
            else:
                frames = []
        if frames:
            self._emit_frames(frames)

    def _emit_frames(self, frames):
        for frame in frames:
            with self._audio_lock:
                if self._cancel_event.is_set() or self.state != State.RUNNING:
                    return
                first = not self._attempt_started
                self._attempt_started = True
            event = dict(self._current_event)
            if first:
                event.update(status="start", text=self._current_event.get(
                    '_tts_display_text', self._current_text))
            self.parent.put_audio_frame(frame, event)
            self._audio_frames_sent += 1

    def _finish_attempt(self, error=None, collect=False):
        """Flush PCM once, optionally returning it for the prefetch queue."""
        with self._audio_lock:
            if self._attempt_end_sent or self._cancel_event.is_set():
                return [] if collect else None
            remainder = self._attempt_remainder
            self._attempt_remainder = np.array([], dtype=np.float32)
            frames = self._attempt_frames
            self._attempt_frames = []
            if remainder.size:
                if len(remainder) < self.chunk:
                    remainder = np.pad(remainder, (0, self.chunk - len(remainder)))
                frames.append(remainder[:self.chunk].copy())
        if collect:
            # A generated segment is not considered played until the playback
            # worker puts its first frame on the media queue.  Keeping the
            # frames private here makes retrying an unplayed segment safe.
            with self._audio_lock:
                self._attempt_end_sent = True
            return frames
        if frames:
            self._emit_frames(frames)
        with self._audio_lock:
            started = self._attempt_started
            if not started or self._attempt_end_sent or self._cancel_event.is_set():
                if error and not self._cancel_event.is_set():
                    self._notify_error(error)
                return [] if collect else None
            self._attempt_end_sent = True
        end_event = dict(self._current_event)
        end_event.update(status="end", text=self._current_event.get(
            '_tts_display_text', self._current_text), _audio_type=1)
        self.parent.put_audio_frame(np.zeros(self.chunk, dtype=np.float32), end_event)
        if error:
            self._notify_error(error)
        return [] if collect else None

    def _notify_error(self, reason):
        sender = getattr(self.parent, 'send_msg', None)
        if sender:
            event = {key: value for key, value in self._current_event.items()
                     if not key.startswith('_')}
            event.update(status='speech_error', error=str(reason),
                         speech_id=self._current_event.get('_utterance_id', ''),
                         dialog_generation=self._current_event.get('_dialog_generation', 0),
                         request_id=self._current_event.get('_request_id', ''))
            try:
                sender(__import__('json').dumps(event, ensure_ascii=False))
            except Exception:
                logger.debug('Unable to notify browser of QwenTTS error', exc_info=True)

    def flush_talk(self):
        self._cancel_event.set()
        self._response_event.set()
        self._attempt_token = uuid.uuid4().hex
        self._close_client()
        super().flush_talk()

    def stop_tts(self):
        self._cancel_event.set()
        self._response_event.set()
        self._attempt_token = uuid.uuid4().hex
        self._close_client()
        self._clear_prefetch_queue()
        logger.info("QwenTTS 已关闭")
