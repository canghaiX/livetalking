import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from avatars.audio_features.base_asr import BaseASR
from avatars.audio_features.mel import MelASR


class MelAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.parent = SimpleNamespace(speech_epoch=0, custom_audiotype=0)
        self.asr = MelASR(SimpleNamespace(fps=25, l=10, r=10, batch_size=16), self.parent)
        self.asr.warm_up()

    def feed(self, count, origin=0):
        for index in range(count):
            samples = np.arange(origin + index * 320, origin + (index + 1) * 320,
                                dtype=np.float32)
            self.asr.put_audio_frame(samples, {'_speech_epoch': self.parent.speech_epoch})

    @staticmethod
    def marked_mel(samples):
        return np.tile(samples[::200], (80, 1))

    def step(self):
        with patch('avatars.audio_features.mel.audio.melspectrogram', self.marked_mel):
            self.asr.run_step()
        return self.asr.feat_queue.get_nowait()

    def assert_aligned(self, batch):
        self.assertEqual(len(batch.features), 16)
        self.assertEqual(len(batch.audio_frames), 32)
        for i, feature in enumerate(batch.features):
            first, second = batch.audio_frames[i * 2:i * 2 + 2]
            self.assertEqual(second.userdata['_audio_sequence'], first.userdata['_audio_sequence'] + 1)
            self.assertEqual(second.userdata['_audio_sample_start'], first.userdata['_audio_sample_start'] + 320)
            # The feature window is tied to the same audio clock. Rounding
            # the 3.2-hop video step keeps the error below one hop.
            error = batch.mel_sample_starts[i] - first.userdata['_audio_sample_start']
            self.assertLessEqual(abs(error), 100)

    def test_numbered_audio_matches_mel_across_multiple_batches(self):
        self.feed(106)
        batches = [self.step() for _ in range(3)]
        for batch in batches:
            self.assert_aligned(batch)
        numbers = [frame.userdata['_audio_sequence']
                   for batch in batches for frame in batch.audio_frames]
        self.assertEqual(numbers, list(range(96)))
        self.assertEqual(batches[0].mel_starts[0], 16)
        self.assertEqual(batches[0].mel_sample_starts[0], 0)
        self.assertEqual(batches[0].mel_sample_ends[0], 3200)
        self.assertEqual(self.asr.output_preroll_frames(), 10)
        self.assertTrue(self.asr.output_queue.empty())

    def test_repeated_interrupts_clear_batches_and_preserve_new_speech(self):
        for epoch in range(3):
            self.parent.speech_epoch = epoch
            self.asr.flush_talk()
            self.assertTrue(self.asr.feat_queue.empty())
            self.feed(42, origin=epoch * 100000)
            batch = self.step()
            self.assertEqual(batch.epoch, epoch)
            self.assert_aligned(batch)
            self.assertEqual(batch.audio_frames[0].data[0], epoch * 100000)
            self.assertTrue(all(frame.userdata['_speech_epoch'] == epoch
                                for frame in batch.audio_frames))

    def test_interrupt_during_extraction_never_enqueues_old_batch(self):
        self.feed(42)
        def interrupt(samples):
            self.parent.speech_epoch += 1
            self.asr.flush_talk()
            return self.marked_mel(samples)
        with patch('avatars.audio_features.mel.audio.melspectrogram', interrupt):
            self.asr.run_step()
        self.assertTrue(self.asr.feat_queue.empty())

    def test_base_asr_keeps_legacy_preroll_default(self):
        asr = BaseASR.__new__(BaseASR)
        asr.stride_left_size = 10
        self.assertEqual(asr.output_preroll_frames(), 10)


if __name__ == '__main__':
    unittest.main()
