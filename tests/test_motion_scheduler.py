import json
import random
from pathlib import Path
import unittest

from avatars.motion_scheduler import MotionScheduler


def config():
    return json.loads((Path(__file__).resolve().parents[1] /
                       'deployment/avatar_motion.json').read_text())


def event(utterance, emotion='neutral', gesture='none', round_id='reply', epoch=0):
    return dict(_utterance_id=utterance, _round_id=round_id, _speech_epoch=epoch,
                emotion=emotion, gesture=gesture)


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = MotionScheduler(config(), rng=random.Random(1))

    def frames(self, count, marker=None, speaking=True):
        return [self.scheduler.step(marker, speaking) for _ in range(count)]

    def test_idle_and_plain_speech_use_01(self):
        idle = self.frames(120, speaking=False)
        self.assertEqual({frame.profile for frame in idle}, {'idle'})
        self.assertTrue(all(0 <= frame.index <= 100 for frame in idle))
        plain = self.frames(200, event('plain'), speaking=True)
        self.assertEqual({frame.profile for frame in plain}, {'smile'})
        self.assertTrue(all(0 <= frame.index <= 100 for frame in plain))

    def test_legacy_introduce_maps_to_03_guide(self):
        first = self.scheduler.step(event('show', gesture='introduce'), True)
        self.assertEqual((first.profile, first.index), ('guide', 0))
        guide_end = self.scheduler.profiles['guide']['end']
        frames = [first] + self.frames(guide_end, event('show', gesture='guide'))
        self.assertEqual([f.index for f in frames if f.profile == 'guide'], list(range(guide_end + 1)))
        self.assertEqual(self.scheduler.profile, 'smile')

    def test_wave_and_emphasis_are_one_shot(self):
        wave = self.scheduler.step(event('wave', gesture='wave'), True)
        self.assertEqual((wave.profile, wave.index), ('wave', 0))
        last = self.scheduler.profiles['wave']['end']
        frames = [wave] + self.frames(last, event('wave', gesture='wave'))
        self.assertEqual([frame.index for frame in frames], list(range(last + 1)))
        self.assertEqual(self.scheduler.profile, 'smile')

        self.scheduler.interrupt(1)
        emphasis = self.scheduler.step(event('emphasis', gesture='emphasis', epoch=1), True)
        self.assertEqual((emphasis.profile, emphasis.index), ('emphasis', 0))

    def test_05_and_06_are_absent_from_all_motion_profiles(self):
        actions = {profile['action_id'] for profile in self.scheduler.profiles.values()}
        self.assertEqual(actions, {'welcome_smile_uniform_v3', 'neutral_explain_uniform_v3',
                                   'closing_thanks_uniform_v3',
                                   'gentle_introduce_07_uniform_v3'})
        self.assertEqual(self.scheduler.profiles['wave']['action_id'],
                         self.scheduler.profiles['emphasis']['action_id'])

    def test_waiting_gesture_only_starts_at_a_new_segment(self):
        self.scheduler.step(event('one', gesture='wave'), True)
        self.frames(160)
        self.scheduler.step(event('two', gesture='emphasis'), True)
        continuing = self.frames(80, event('two', gesture='emphasis'))
        self.assertTrue(all(frame.profile == 'smile' for frame in continuing))
        next_segment = self.scheduler.step(event('three'), True)
        self.assertEqual(next_segment.profile, 'emphasis')

    def test_long_guide_returns_to_baseline_before_another_gesture(self):
        self.scheduler.step(event('guide', gesture='guide'), True)
        self.frames(99)
        self.scheduler.step(event('pending', gesture='wave'), True)
        self.frames(149)
        self.assertEqual(self.scheduler.profile, 'smile')
        # The shortened 03 action leaves a baseline interval before the
        # eight-second gesture cooldown expires; the next segment may then
        # start the queued wave.
        self.assertEqual(self.scheduler.step(event('after-normal-speech'), True).profile, 'wave')

    def test_round_end_discards_a_waiting_gesture(self):
        self.scheduler.step(event('one', gesture='wave'), True)
        self.scheduler.step(event('two', gesture='emphasis'), True)
        ending = dict(event('two'), status='end', _round_final=True)
        self.scheduler.step(ending, True, audio_events=[ending])
        self.assertIsNone(self.scheduler.waiting_gesture)
        self.frames(220, speaking=False)
        self.assertEqual(self.scheduler.profile, 'idle')

    def test_silence_does_not_replace_continuous_baseline_speech(self):
        self.scheduler.step(event('one', gesture='wave'), True)
        self.frames(149)
        self.frames(250, speaking=False)
        self.assertEqual(self.scheduler.step(event('two', gesture='wave'), True).profile, 'smile')
        self.frames(24, event('two', gesture='wave'))
        self.assertEqual(self.scheduler.step(event('three'), True).profile, 'wave')

    def test_serious_emotion_keeps_01_and_never_uses_02(self):
        frame = self.scheduler.step(event('safety', emotion='serious'), True)
        self.assertEqual(frame.profile, 'smile')
        self.assertEqual(self.scheduler.profiles['serious']['action_id'],
                         'welcome_smile_uniform_v3')

    def test_gesture_count_and_cooldown(self):
        first = self.scheduler.step(event('one', gesture='wave'), True,
                                    audio_events=[dict(event('one', gesture='wave'),
                                                       _audio_sample_start=0)])
        self.assertEqual(first.profile, 'wave')
        self.scheduler.played(first.started, 0, 0)
        self.frames(199, speaking=True)
        blocked = self.scheduler.step(event('two', gesture='emphasis', round_id='two'), True,
                                      audio_events=[dict(event('two', gesture='emphasis', round_id='two'),
                                                         _audio_sample_start=127999)])
        self.assertEqual(blocked.profile, 'smile')
        allowed = self.scheduler.step(event('three', gesture='emphasis', round_id='three'), True,
                                      audio_events=[dict(event('three', gesture='emphasis', round_id='three'),
                                                         _audio_sample_start=128000)])
        self.assertEqual(allowed.profile, 'emphasis')

    def test_three_gestures_per_round_and_fourth_is_rejected(self):
        starts = []
        for i, gesture in enumerate(['wave', 'emphasis', 'wave', 'emphasis']):
            frame = self.scheduler.step(event(f'segment-{i}', gesture=gesture), True)
            if frame.started:
                starts.append((self.scheduler.frame - 1, frame.profile))
            self.frames(199, speaking=True)
        self.assertEqual(len(starts), 3)
        self.assertEqual(self.scheduler.gesture_rounds['reply'], 3)
        self.assertIsNone(self.scheduler.waiting_gesture)

    def test_closing_waits_for_final_silence_then_returns_to_idle(self):
        marker = event('thanks', 'closing')
        marker.update(_close_after_audio=True, status='end', _round_final=True)
        self.assertEqual(self.scheduler.step(marker, True).profile, 'smile')
        frames = self.frames(300, speaking=False)
        self.assertEqual([f.index for f in frames if f.profile == 'closing'], list(range(250)))
        self.assertEqual(frames[-1].profile, 'idle')

    def test_interrupt_cancels_pending_gestures_and_closing(self):
        self.scheduler.step(event('one', gesture='wave'), True)
        self.scheduler.step(event('two', gesture='emphasis'), True)
        self.scheduler.interrupt(1)
        self.assertIsNone(self.scheduler.waiting_gesture)
        self.assertIsNone(self.scheduler.pending_closing)
        self.assertEqual(self.scheduler.step(event('old', gesture='wave'), True).profile, 'smile')
        self.assertEqual(self.scheduler.step(event('new', gesture='wave', epoch=1), True).profile, 'wave')

    def test_missing_profile_falls_back_to_smile(self):
        self.scheduler.disable('wave')
        self.assertEqual(self.scheduler.step(event('show', gesture='wave'), True).profile, 'smile')

    def test_long_reply_can_nod_once_at_segment_boundary_without_stopping_speech(self):
        scheduler = MotionScheduler(config(), rng=random.Random(1))
        scheduler.step(dict(event('one', round_id='long'), text='第一段。'), True)
        scheduler.step(dict(event('two', round_id='long'), text='第二段内容。'), True)
        frame = scheduler.step(dict(event('three', round_id='long'), text='第三段内容。'), True)
        self.assertEqual((frame.profile, frame.index), ('nod', 0))
        self.assertTrue(scheduler.oneshot)
        for _ in range(249):
            scheduler.step({}, True)
        self.assertEqual(scheduler.profile, 'smile')
        scheduler.step(dict(event('four', round_id='long'), text='第四段。'), True)
        self.assertNotEqual(scheduler.profile, 'nod')

    def test_short_reply_does_not_nod(self):
        scheduler = MotionScheduler(config(), rng=random.Random(0))
        scheduler.step(dict(event('one', round_id='short'), text='短。'), True)
        scheduler.step(dict(event('two', round_id='short'), text='短。'), True)
        self.assertNotEqual(scheduler.step(dict(event('three', round_id='short'), text='短。'), True).profile, 'nod')


if __name__ == '__main__':
    unittest.main()
