"""Per-session motion playback, independent of inference batch boundaries."""

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import time
import random


GESTURE_ALIASES = {
    # Keep replies produced by older prompts playable after the profile rename.
    'introduce': 'guide',
    'guide': 'guide',
    'wave': 'wave',
    'emphasis': 'emphasis',
    'none': 'none',
}


@dataclass(frozen=True)
class MotionFrame:
    profile: str
    index: int
    epoch: int
    started: str | None = None


class MotionScheduler:
    def __init__(self, config, fps=25, clock=time.monotonic, rng=None):
        self.profiles = dict(config['profiles'])
        if 'smile' not in self.profiles:
            raise ValueError('Smile profile is required')
        self.idle_profile = config.get('idle_profile', 'smile')
        if self.idle_profile not in self.profiles:
            raise ValueError('Idle profile is not configured')
        for name, profile in self.profiles.items():
            if (not isinstance(profile['start'], int) or not isinstance(profile['end'], int)
                    or profile['start'] < 0 or profile['end'] < profile['start']):
                raise ValueError(f'Invalid frame range for {name}')
            idle = profile.get('idle', profile['start'])
            if not isinstance(idle, int) or not profile['start'] <= idle <= profile['end']:
                raise ValueError(f'Invalid idle frame for {name}')
            for phase in ('enter', 'exit'):
                phase_start = profile.get(f'{phase}_start')
                phase_end = profile.get(f'{phase}_end')
                if phase_start is None and phase_end is None:
                    continue
                if (not isinstance(phase_start, int) or not isinstance(phase_end, int)
                        or phase_start < 0 or phase_end < phase_start):
                    raise ValueError(f'Invalid {phase} frame range for {name}')
        self.fps = fps
        self.clock = clock
        self.rng = rng if rng is not None else random.Random()
        self.hold_frames = round(config.get('expression_hold_seconds', 4) * fps)
        self.cooldown = config.get('gesture_cooldown_seconds', 8)
        self.nod_cooldown = config.get('nod_cooldown_seconds', 24)
        self.nod_probability = float(config.get('nod_probability', 0.35))
        self.max_nods = int(config.get('max_nods_per_round', 1))
        self.nod_min_segments = int(config.get('nod_min_segments', 3))
        self.nod_min_chars = int(config.get('nod_min_chars', 80))
        self.max_gestures = config.get('max_gestures_per_round', 3)
        self.idle_frames = round(config.get('idle_return_seconds', .6) * fps)
        self.transition_frames = config.get('transition_frames', 6)
        if (self.hold_frames < 0 or self.cooldown < 0 or self.nod_cooldown < 0
                or not 0 <= self.nod_probability <= 1 or self.max_nods < 0
                or self.nod_min_segments < 1 or self.nod_min_chars < 0 or self.idle_frames < 1
                or not isinstance(self.max_gestures, int) or self.max_gestures < 0
                or not isinstance(self.transition_frames, int) or self.transition_frames < 1):
            raise ValueError('Invalid motion timing configuration')
        self.lock = RLock()
        self.epoch = 0
        self.last_gesture_played = None
        self.pending_gesture = None
        self.waiting_gesture = None
        self.last_gesture_frame = None
        self.media_sample_position = None
        self.last_gesture_media_sample = None
        self.baseline_speech_frames = 0
        self.gesture_tokens = OrderedDict()
        self.ended_rounds = OrderedDict()
        self.seen = OrderedDict()
        self.gesture_rounds = OrderedDict()
        self.nod_rounds = OrderedDict()
        self.round_segments = OrderedDict()
        self.round_chars = OrderedDict()
        self.last_nod_frame = None
        self.last_nod_media_sample = None
        self.closing_rounds = OrderedDict()
        self.pending_closing = None
        self.closing_ready = False
        self.current_round = None
        self.frame = 0
        self._reset_playback()

    def _reset_playback(self):
        self.expression = 'smile'
        self.profile = self.idle_profile
        self.cursor = 0
        self.expression_since = self.frame - self.hold_frames
        self.oneshot = False
        self.deferred = None
        self.phase = None
        self.phase_cursor = 0
        self.pending_expression = None
        self.silent_frames = 0

    @staticmethod
    def _remember(cache, key):
        cache[key] = True
        while len(cache) > 128:
            cache.popitem(last=False)

    def interrupt(self, epoch):
        with self.lock:
            self.epoch = epoch
            self.pending_gesture = None
            self.waiting_gesture = None
            self.pending_closing = None
            self.closing_ready = False
            self.seen.clear()
            self.gesture_rounds.clear()
            self.nod_rounds.clear()
            self.round_segments.clear()
            self.round_chars.clear()
            self.ended_rounds.clear()
            self.current_round = None
            self.last_gesture_frame = None
            self.last_gesture_media_sample = None
            self.last_gesture_played = None
            self.last_nod_frame = None
            self.last_nod_media_sample = None
            self.baseline_speech_frames = 0
            self._reset_playback()

    def disable(self, profile):
        with self.lock:
            if profile != 'smile':
                self.profiles.pop(profile, None)
            if self.profile == profile:
                self.pending_gesture = None
                self._reset_playback()
            if self.waiting_gesture and self.waiting_gesture[0] == profile:
                self.waiting_gesture = None

    def played(self, token, epoch, media_sample_start=None):
        """Called when WebRTC consumes the first audio frame of the gesture."""
        with self.lock:
            if epoch == self.epoch and token in self.gesture_tokens:
                self.last_gesture_played = self.clock()
                if media_sample_start is not None:
                    self.last_gesture_media_sample = int(media_sample_start)
                if token == self.pending_gesture:
                    self.pending_gesture = None
                if token.startswith('nod:'):
                    self.last_nod_frame = self.frame
                    if media_sample_start is not None:
                        self.last_nod_media_sample = int(media_sample_start)

    def _nod_allowed(self):
        if self.last_nod_media_sample is not None and self.media_sample_position is not None:
            return self.media_sample_position - self.last_nod_media_sample >= round(self.nod_cooldown * 16000)
        return (self.last_nod_frame is None or
                self.frame - self.last_nod_frame >= round(self.nod_cooldown * self.fps))

    def _try_start_nod(self, round_id, speaking):
        enough = (self.round_segments.get(round_id, 0) >= self.nod_min_segments or
                  self.round_chars.get(round_id, 0) >= self.nod_min_chars)
        if (not speaking or self.oneshot or 'nod' not in self.profiles or not enough
                or self.nod_rounds.get(round_id, 0) >= self.max_nods
                or not self._nod_allowed() or self.rng.random() >= self.nod_probability):
            return None
        self.nod_rounds[round_id] = self.nod_rounds.get(round_id, 0) + 1
        token = f'nod:{self.epoch}:{round_id}:{self.nod_rounds[round_id]}'
        self._remember(self.gesture_tokens, token)
        self.last_nod_frame = self.frame
        self.last_nod_media_sample = self.media_sample_position
        self._start('nod')
        return token

    def _try_start_waiting_gesture(self, speaking):
        if (not speaking or self.oneshot or not self.waiting_gesture
                or self.current_round in self.ended_rounds):
            return None
        if self.last_gesture_frame is not None and self.baseline_speech_frames < self.fps:
            return None
        if self.media_sample_position is not None and self.last_gesture_media_sample is not None:
            if self.media_sample_position - self.last_gesture_media_sample < round(self.cooldown * 16000):
                return None
        elif (self.last_gesture_frame is not None
              and self.frame - self.last_gesture_frame < round(self.cooldown * self.fps)):
            return None
        gesture, round_id, utterance = self.waiting_gesture
        gesture = GESTURE_ALIASES.get(gesture, 'none')
        if round_id != self.current_round or gesture not in self.profiles:
            return None
        count = self.gesture_rounds.get(round_id, 0)
        if count >= self.max_gestures:
            self.waiting_gesture = None
            return None
        self.waiting_gesture = None
        self.gesture_rounds[round_id] = count + 1
        while len(self.gesture_rounds) > 128:
            self.gesture_rounds.popitem(last=False)
        token = f'{self.epoch}:{utterance}'
        self._remember(self.gesture_tokens, token)
        self.pending_gesture = token
        self.last_gesture_frame = self.frame
        self.baseline_speech_frames = 0
        self._start(gesture)
        return token

    def _set_expression(self, name, force=False):
        if name not in self.profiles:
            name = 'smile'
        if name != self.expression:
            if not force and self.frame - self.expression_since < self.hold_frames:
                return
            old = self.expression
            old_profile = self.profiles.get(old, {})
            exit_start = old_profile.get('exit_start')
            exit_end = old_profile.get('exit_end')
            if (not self.oneshot and exit_start is not None and exit_end is not None):
                self.pending_expression = name
                self.phase = 'exit'
                self.phase_cursor = 0
                self.oneshot = True
                return
            self.expression = name
            self.expression_since = self.frame
            enter_start = self.profiles[name].get('enter_start')
            enter_end = self.profiles[name].get('enter_end')
            if not self.oneshot and enter_start is not None and enter_end is not None:
                self.phase = 'enter'
                self.phase_cursor = 0
                self.oneshot = True
                self.profile = name
                return
        if not self.oneshot and self.profile != name:
            self.profile = name
            self.cursor = 0

    def _start(self, profile):
        self.oneshot = True
        self.phase = 'oneshot'
        self.profile = profile
        self.cursor = 0

    def _finish_oneshot(self):
        finished = self.profile
        self.oneshot = False
        self.phase = None
        self.phase_cursor = 0
        if finished in {'closing', 'nod'}:
            self._set_expression('smile', force=True)
            if self.oneshot:
                return
        elif self.deferred:
            desired = self.deferred
            self.deferred = None
            self._set_expression(desired, force=True)
            if self.oneshot:
                return
        if self.pending_expression:
            desired = self.pending_expression
            self.pending_expression = None
            self.expression = desired
            self.expression_since = self.frame
            enter_start = self.profiles[desired].get('enter_start')
            enter_end = self.profiles[desired].get('enter_end')
            if enter_start is not None and enter_end is not None:
                self.phase = 'enter'
                self.phase_cursor = 0
                self.oneshot = True
                self.profile = desired
                self.cursor = 0
                return
        self.profile = self.expression if self.expression in self.profiles else 'smile'
        self.cursor = 0

    def step(self, event=None, speaking=False, audio_events=None):
        with self.lock:
            event = event or {}
            started = None
            for marker in (audio_events or [event]):
                if marker.get('_audio_sample_start') is not None:
                    self.media_sample_position = int(marker['_audio_sample_start'])
                    break
            self.silent_frames = 0 if speaking else self.silent_frames + 1
            if speaking and not self.oneshot and self.profile in {'smile', self.idle_profile}:
                self.baseline_speech_frames += 1
            else:
                self.baseline_speech_frames = 0
            utterance = event.get('_utterance_id')
            valid = event.get('_speech_epoch', self.epoch) == self.epoch
            if valid and utterance and utterance not in self.seen:
                self._remember(self.seen, utterance)
                round_id = event.get('_round_id', utterance)
                if round_id != self.current_round:
                    self.waiting_gesture = None
                    self.pending_closing = None
                    self.closing_ready = False
                    self.round_segments[round_id] = 0
                    self.round_chars[round_id] = 0
                self.current_round = round_id
                self.round_segments[round_id] = self.round_segments.get(round_id, 0) + 1
                display_text = event.get('_tts_display_text', event.get('text', ''))
                self.round_chars[round_id] = self.round_chars.get(round_id, 0) + len(display_text or '')
                is_final_closing = (event.get('emotion') == 'closing'
                                    and event.get('_close_after_audio')
                                    and 'closing' in self.profiles)
                if is_final_closing and round_id not in self.closing_rounds:
                    # Queue the closing independently of any gesture or
                    # expression transition that is already in progress.
                    self.pending_closing = round_id
                    self.waiting_gesture = None
                gesture = GESTURE_ALIASES.get(event.get('gesture'), 'none')
                if (not is_final_closing and gesture in {'guide', 'wave', 'emphasis'}
                        and gesture in self.profiles and round_id not in self.ended_rounds
                        and self.gesture_rounds.get(round_id, 0) < self.max_gestures):
                    self.waiting_gesture = (gesture, round_id, utterance)
                # 02 is intentionally disabled. Serious content keeps the
                # stable 01 presentation pose and does not reset the avatar.
                desired = 'smile'
                started = self._try_start_waiting_gesture(
                    speaking and not is_final_closing)
                if (started is None and not is_final_closing and gesture == 'none'
                        and not self.waiting_gesture and not self.oneshot):
                    started = self._try_start_nod(round_id, speaking)
                if self.oneshot:
                    self.deferred = desired
                elif not is_final_closing:
                    self._set_expression(desired)

            # A closing gesture is queued by the final segment, but starts
            # only after its end marker has reached a silent audio pair.
            for marker in (audio_events or [event]):
                if (marker.get('_speech_epoch', self.epoch) == self.epoch
                        and marker.get('status') == 'end'
                        and marker.get('_round_final')):
                    round_id = marker.get('_round_id', marker.get('_utterance_id'))
                    if round_id and round_id == self.current_round:
                        self._remember(self.ended_rounds, round_id)
                        self.waiting_gesture = None
                        if marker.get('_close_after_audio') and round_id not in self.closing_rounds:
                            self.pending_closing = round_id
                            self.closing_ready = True
            if (self.pending_closing and self.closing_ready and not speaking and not self.oneshot):
                self._remember(self.closing_rounds, self.pending_closing)
                self.pending_closing = None
                self.closing_ready = False
                self._start('closing')

            if not self.oneshot and self.silent_frames >= self.idle_frames:
                if self.expression != 'smile':
                    self.expression = 'smile'
                    self.expression_since = self.frame

            if not self.oneshot:
                if speaking and self.profile == self.idle_profile:
                    self.profile = self.expression if self.expression in self.profiles else 'smile'
                    self.cursor = 0
                elif not speaking and self.silent_frames >= self.idle_frames:
                    self.profile = self.idle_profile

            profile = self.profiles[self.profile]
            if self.oneshot and self.phase in {'enter', 'exit'}:
                start = profile[f'{self.phase}_start']
                end = profile[f'{self.phase}_end']
                cursor = self.phase_cursor
            else:
                start, end = profile['start'], profile['end']
                cursor = self.cursor
            length = end - start + 1
            if self.oneshot:
                index = start + cursor
                advance_cursor = True
            elif not speaking and self.profile == self.idle_profile:
                # Play a dedicated closed-mouth waiting loop while no audio is being played.
                turn, offset = divmod(self.cursor, length)
                index = start + (offset if turn % 2 == 0 else length - offset - 1)
                advance_cursor = True
            else:
                turn, offset = divmod(self.cursor, length)
                index = start + (offset if turn % 2 == 0 else length - offset - 1)
                advance_cursor = True
            result = MotionFrame(self.profile, index, self.epoch, started)
            if advance_cursor:
                if self.oneshot and self.phase in {'enter', 'exit'}:
                    self.phase_cursor += 1
                else:
                    self.cursor += 1
            self.frame += 1
            phase_done = (self.oneshot and self.phase in {'enter', 'exit'}
                          and self.phase_cursor == length)
            action_done = (self.oneshot and self.phase == 'oneshot'
                           and self.cursor == length)
            if phase_done:
                if self.phase == 'enter':
                    # Re-run deferred expression handling after the entry
                    # segment. This lets a serious->smile request arriving
                    # during entry play the serious exit segment instead of
                    # being silently dropped.
                    self._finish_oneshot()
                else:
                    self._finish_oneshot()
            elif action_done:
                self._finish_oneshot()
            return result
