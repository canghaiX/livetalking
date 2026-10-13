import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from server.avatar_package_manager import PackageManager, validate_id
from server.avatar_video_validation import VideoValidationError, probe_video, validate_faces


class Detector:
    def __init__(self, mode='single', missing=()):
        self.mode = mode
        self.missing = set(missing)
        self.index = 0

    def detect_from_batch(self, batch):
        results = []
        for _ in batch:
            face = [400, 180, 640, 480, .99]
            faces = [face] if self.mode == 'single' else []
            if self.mode == 'multiple':
                faces = [face, [800, 180, 1040, 480, .99]]
            if self.mode == 'small':
                faces = [[400, 180, 500, 480, .99]]
            if self.index in self.missing:
                faces = []
            results.append(faces)
            self.index += 1
        return results


class VideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for name, fps, duration, size in [('valid',25,4,'1280x720'), ('fps',24,4,'1280x720'),
                                          ('low',25,4,'640x360'), ('long',25,16,'1280x720')]:
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'color=c=gray:s={size}:r={fps}',
                            '-t',str(duration),'-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p',
                            str(cls.root / f'{name}.mp4')],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_metadata_valid(self):
        report = probe_video(self.root / 'valid.mp4')
        self.assertEqual(report['frame_count'], 100)
        self.assertEqual(report['fps'], 25)

    def test_metadata_rejections(self):
        for name in ('fps','low','long'):
            with self.subTest(name=name), self.assertRaises(VideoValidationError):
                probe_video(self.root / f'{name}.mp4')
        with self.assertRaises(VideoValidationError):
            probe_video(self.root / 'valid.mp4', [1920,1080])

    def test_variable_timestamps_rejected(self):
        metadata = {'streams':[dict(codec_type='video',codec_name='h264',width=1280,height=720,
                                   avg_frame_rate='25/1',r_frame_rate='25/1',duration='4')],
                    'format':dict(format_name='mov,mp4',duration='4'),
                    'frames':[dict(media_type='video',best_effort_timestamp_time=str(i*.04)) for i in range(100)]}
        metadata['frames'][40]['best_effort_timestamp_time'] = '1.61'
        with patch('server.avatar_video_validation.subprocess.run', return_value=type('Result',(),{'stdout':json.dumps(metadata)})()):
            with self.assertRaises(VideoValidationError):
                probe_video(self.root / 'valid.mp4')

    def test_faces(self):
        path = self.root / 'valid.mp4'
        report = probe_video(path)
        boxes = validate_faces(path, report, Detector(missing=(20,)))
        self.assertEqual(len(boxes),100)
        self.assertEqual(report['repaired_frames'],[20])
        for mode in ('none','multiple','small'):
            with self.subTest(mode=mode), self.assertRaises(VideoValidationError):
                validate_faces(path, probe_video(path), Detector(mode))
        with self.assertRaises(VideoValidationError):
            validate_faces(path, probe_video(path), Detector(missing=(20,30,40)))

    def test_generate_atomic_and_failed_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = PackageManager(folder)
            manager.create('railway','测试角色',[dict(id='neutral_explain',label='中性')])
            source = manager.root / 'railway' / 'source' / 'neutral_explain.mp4'
            source.parent.mkdir(parents=True)
            source.write_bytes((self.root / 'valid.mp4').read_bytes())
            weights = Path(folder) / 'weights.pth'; weights.touch()
            with patch.dict(os.environ,LIVETALKING_SFD_PATH=str(weights)), patch(
                'avatars.wav2lip.face_detection.detection.sfd.sfd_detector.SFDDetector', return_value=Detector()
            ):
                manager.generate('railway','neutral_explain', lambda _:None)
            output = manager.avatars / 'railway__neutral_explain'
            self.assertEqual(len(list((output/'face_imgs').glob('*.png'))),100)
            self.assertTrue((output/'coords.pkl').is_file())
            self.assertEqual(manager.get('railway')['actions'][0]['status'],'completed')
            self.assertFalse(list(manager.avatars.glob('.package-*')))
            manager.publish('railway')
            self.assertEqual(PackageManager(folder).active_avatar(),'railway__neutral_explain')
            self.assertTrue(json.loads((manager.root/'railway'/'manifest.json').read_text())['published'])
            with self.assertRaises(ValueError):
                manager.reserve_upload('railway','neutral_explain')
            manager.create('bad','错误素材',[dict(id='neutral_explain',label='中性')])
            bad_source = manager.root/'bad'/'source'/'neutral_explain.mp4'
            bad_source.parent.mkdir(parents=True); bad_source.write_bytes((self.root/'fps.mp4').read_bytes())
            with self.assertRaises(VideoValidationError):
                manager.generate('bad','neutral_explain',lambda _:None)
            self.assertFalse((manager.avatars/'bad__neutral_explain').exists())
            self.assertEqual(manager.get('bad')['actions'][0]['status'],'failed')


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = PackageManager(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_ids(self):
        for value in ('../x','/tmp/x','a/b','a__b','a.b','',None):
            with self.subTest(value=value), self.assertRaises(ValueError): validate_id(value)
        self.assertEqual(validate_id('railway-v1'),'railway-v1')

    def test_create_and_publish_incomplete(self):
        package = self.manager.create('railway','讲解员')
        self.assertEqual(len(package['actions']),5)
        with self.assertRaises(ValueError): self.manager.create('railway','重复')
        with self.assertRaises(ValueError): self.manager.publish('railway')
        with self.assertRaises(ValueError): self.manager.create('wrong','错误',[])

    def test_upload_reservation_and_restart_recovery(self):
        self.manager.create('railway','讲解员')
        self.manager.reserve_upload('railway','neutral_explain')
        with self.assertRaises(ValueError): self.manager.reserve_upload('railway','neutral_explain')
        restarted = PackageManager(self.temp.name)
        self.assertEqual(restarted.get('railway')['actions'][0]['status'],'failed')
        restarted.reserve_upload('railway','neutral_explain')
        with self.assertRaises(ValueError): restarted.preview_allowed('railway__neutral_explain')

    def test_shared_serial_worker(self):
        import threading
        from server.task_manager import TaskManager
        manager = TaskManager(max_workers=1)
        gate = threading.Event(); done = threading.Event(); order = []
        def first(progress):
            order.append('first'); gate.wait(5); progress(100)
        def second(progress):
            order.append('second'); progress(100); done.set()
        one = manager.add_task('wav2lip','one',{},runner=first)
        two = manager.add_task('wav2lip','two',{},runner=second)
        gate.set(); self.assertTrue(done.wait(5)); manager.executor.shutdown(wait=True)
        self.assertEqual(order,['first','second'])
        self.assertEqual(manager.get_task(one).status,'completed')
        self.assertEqual(manager.get_task(two).progress,100)


if __name__ == '__main__':
    unittest.main()
