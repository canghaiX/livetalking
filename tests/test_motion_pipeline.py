from threading import Lock
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from avatars.base_avatar import AudioFrameData, BaseAvatar
from avatars.wav2lip_avatar import LipReal
from avatars.motion_scheduler import MotionScheduler
from test_motion_scheduler import config, event


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.avatar = LipReal.__new__(LipReal)
        self.avatar.batch_size = 16
        self.avatar.speech_epoch = 0
        settings = config()
        self.action_ids = {name: profile['action_id'] for name, profile in settings['profiles'].items()}
        self.payloads = {}
        for profile in settings['profiles'].values():
            profile['avatar_id'] = profile['action_id']
            self.payloads[profile['avatar_id']] = ([None] * 250, [None] * 250, [None] * 250)
        self.avatar.default_avatar_payload = self.payloads[settings['profiles']['smile']['action_id']]
        self.avatar.motion_scheduler = MotionScheduler(settings)
        self.avatar.avatar_payload_for = self.payloads.__getitem__
        self.avatar.inference_batch = lambda index, features, avatar_payload, frame_indices: frame_indices

    def batch(self, marker, speaking=True):
        audio = [AudioFrameData(np.zeros(320), 0 if speaking else 1, marker) for _ in range(32)]
        return self.avatar.infer_motion_batch([None] * 16, audio)

    def test_frame_selection_remains_consistent_at_gesture_end(self):
        marker = event('show', gesture='guide')
        output = [item for _ in range(17) for item in self.batch(marker)]
        gesture = [item for item in output if item[4]['profile'] == 'guide']
        guide_end = config()['profiles']['guide']['end']
        self.assertEqual([item[2] for item in gesture], list(range(guide_end + 1)))
        self.assertTrue(all(prediction == index for prediction, _, index, _, _ in output))
        self.assertTrue(all(item[3] is self.payloads[self.action_ids['guide']] for item in gesture))
        self.assertEqual((output[guide_end + 1][4]['profile'], output[guide_end + 1][2]), ('smile', 0))
        markers = [frame.userdata.get('_motion_started') for item in output for frame in item[1]]
        self.assertEqual(sum(bool(marker) for marker in markers), 1)

    def test_stale_audio_is_silenced_after_interrupt(self):
        marker = event('old', gesture='guide')
        self.avatar.speech_epoch = 1
        self.avatar.motion_scheduler.interrupt(1)
        output = self.batch(marker)
        self.assertTrue(all(item[0] is None and item[4]['profile'] == 'idle' for item in output))
        self.assertTrue(all(frame.type == 1 for item in output for frame in item[1]))

    def test_failed_material_returns_to_default(self):
        def missing(avatar_id):
            if avatar_id == self.action_ids['guide']:
                raise FileNotFoundError('missing gesture')
            return self.payloads[avatar_id]
        self.avatar.avatar_payload_for = missing
        output = self.batch(event('show', gesture='guide'))
        self.assertTrue(all(item[4]['profile'] == 'smile' for item in output))
        self.assertTrue(all(item[4]['avatar_id'] == self.action_ids['smile'] for item in output))
        self.assertIsNone(self.avatar.motion_scheduler.pending_gesture)

    def test_missing_07_never_falls_back_to_05_or_06(self):
        def missing(avatar_id):
            if avatar_id == self.action_ids['wave']:
                raise FileNotFoundError('missing 07')
            return self.payloads[avatar_id]
        self.avatar.avatar_payload_for = missing
        for gesture in ('wave', 'emphasis'):
            output = self.batch(event(gesture, gesture=gesture))
            self.assertTrue(all(item[4]['avatar_id'] == self.action_ids['smile'] for item in output))

    def test_inference_uses_explicit_face_indices(self):
        avatar = LipReal.__new__(LipReal)
        faces = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(12)]
        captured = []
        def model(features, images):
            captured.extend(images[:, 3, 0, 0].cpu().tolist())
            return torch.zeros(len(images), 3, 4, 4)
        avatar.model = model
        with patch('avatars.wav2lip_avatar.device', 'cpu'):
            avatar.inference_batch(0, [np.zeros((80, 16))] * 3,
                                   avatar_payload=([None] * 12, faces, [None] * 12),
                                   frame_indices=[9, 2, 11])
        np.testing.assert_allclose(captured, np.array([9, 2, 11]) / 255, rtol=1e-6)

    def test_motion_metadata_contains_action_id_for_transition_identity(self):
        output = self.batch(event('plain'), speaking=True)
        self.assertTrue(output)
        self.assertEqual(output[0][4]['avatar_id'], self.action_ids['smile'])

    def test_delayed_llm_reply_cannot_enqueue_after_interrupt(self):
        avatar = BaseAvatar.__new__(BaseAvatar)
        avatar.speech_epoch = 1
        avatar.speech_lock = Lock()
        avatar.tts = SimpleNamespace(put_msg_txt=lambda text, event: queued.put(event))
        queued = Queue()
        avatar.put_msg_txt('old reply', {'_speech_epoch': 0})
        self.assertTrue(queued.empty())
        avatar.put_msg_txt('new reply')
        metadata = queued.get_nowait()
        self.assertEqual(metadata['_speech_epoch'], 1)
        self.assertTrue(metadata['_utterance_id'])

    def test_playback_cooldown_uses_the_current_epoch_audio_clock(self):
        avatar = BaseAvatar.__new__(BaseAvatar)
        avatar.speech_epoch = 2
        avatar.sessionid = 'cooldown'
        calls = []
        avatar.motion_scheduler = SimpleNamespace(played=lambda *args: calls.append(args))
        avatar.notify({'_speech_epoch': 2, '_motion_started': 'gesture',
                       '_audio_sample_start': 3200, '_media_sample_start': 963200})
        self.assertEqual(calls, [('gesture', 2, 3200)])


if __name__ == '__main__':
    unittest.main()
