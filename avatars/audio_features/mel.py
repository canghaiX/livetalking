###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Mel 频谱音频特征提取 — 用于 Wav2Lip
#  迁移自 lipasr.py
#

import time
import torch
import numpy as np

import queue
from queue import Queue
from dataclasses import dataclass
from threading import RLock

from avatars.audio_features.base_asr import BaseASR
from avatars.base_avatar import AudioFrameData
from avatars.wav2lip import audio
from avatars.wav2lip.hparams import hparams


@dataclass(frozen=True)
class MelBatch:
    features: list
    audio_frames: list
    epoch: int
    mel_starts: tuple
    mel_sample_starts: tuple
    mel_sample_ends: tuple

class MelASR(BaseASR):
    paired_batches = True

    def __init__(self, opt, parent=None):
        super().__init__(opt, parent)
        self._reset_lock = RLock()
        self._reset_requested = True
        self._epoch = getattr(parent, 'speech_epoch', 0)
        self._context = []
        self._next_sequence = 0

    def output_preroll_frames(self):
        # l/r count audio chunks (20ms), not video frames (40ms).
        return self.stride_left_size

    def warm_up(self):
        # Context is initialized by the extraction thread, including after an
        # interrupt. No live speech is discarded to initialize the left side.
        self._reset_requested = True

    def flush_talk(self):
        with self._reset_lock:
            self._reset_requested = True
            for target in (self.queue, self.feat_queue):
                with target.mutex:
                    target.queue.clear()
                    target.not_full.notify_all()

    def _initialize_context(self, epoch):
        self._epoch = epoch
        self._next_sequence = 0
        self._context = [AudioFrameData(
            np.zeros(self.chunk, dtype=np.float32), 1,
            {'_speech_epoch': epoch, '_audio_sequence': i,
             '_audio_sample_start': i * self.chunk})
            for i in range(-self.stride_left_size, 0)]

    def run_step(self):
        with self._reset_lock:
            epoch = getattr(self.parent, 'speech_epoch', self._epoch)
            if self._reset_requested or epoch != self._epoch:
                self._initialize_context(epoch)
                self._reset_requested = False
        count = self.batch_size * 2
        required = self.stride_left_size + count + self.stride_right_size
        while len(self._context) < required:
            frame = self.get_audio_frame()
            if frame.userdata.get('_speech_epoch', epoch) != epoch:
                continue
            metadata = dict(frame.userdata, _speech_epoch=epoch,
                            _audio_sequence=self._next_sequence,
                            _audio_sample_start=self._next_sequence * self.chunk)
            self._context.append(AudioFrameData(frame.data, frame.type, metadata))
            self._next_sequence += 1
            with self._reset_lock:
                if self._reset_requested or getattr(self.parent, 'speech_epoch', epoch) != epoch:
                    return

        inputs = np.concatenate([frame.data for frame in self._context])
        mel = audio.melspectrogram(inputs)
        hop = hparams.hop_size
        # A 40ms video step is 3.2 Mel hops.  Rounding the cumulative time
        # keeps each feature window tied to the same audio sample clock;
        # flooring every step would accumulate almost one video frame of
        # drift over a 16-frame batch.
        left_samples = self.stride_left_size * self.chunk
        starts = tuple(round((left_samples + i * 2 * self.chunk) / hop)
                       for i in range(self.batch_size))
        features = [mel[:, start:start + 16] for start in starts]
        if any(feature.shape != (80, 16) for feature in features):
            raise ValueError('Insufficient right audio context for Wav2Lip')
        paired_audio = self._context[self.stride_left_size:self.stride_left_size + count]
        context_sample_start = self._context[0].userdata.get('_audio_sample_start', -left_samples)
        mel_sample_starts = tuple(context_sample_start + start * hop for start in starts)
        mel_sample_ends = tuple(start + 16 * hop for start in mel_sample_starts)
        batch = MelBatch(features, paired_audio, epoch, starts,
                         mel_sample_starts, mel_sample_ends)
        keep = self.stride_left_size + self.stride_right_size
        self._context = self._context[-keep:] if keep else []
        while True:
            with self._reset_lock:
                if self._reset_requested or getattr(self.parent, 'speech_epoch', epoch) != epoch:
                    return
                try:
                    self.feat_queue.put_nowait(batch)
                    return
                except queue.Full:
                    pass
            time.sleep(.005)
