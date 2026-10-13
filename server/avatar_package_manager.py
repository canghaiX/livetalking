"""Persistent character packages; generation shares the existing serial task worker."""
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from server.avatar_video_validation import VideoValidationError, probe_video, validate_faces


DEFAULT_ACTIONS = [
    ('neutral_explain', '中性讲解'), ('welcome_smile', '欢迎微笑'),
    ('serious_emphasis', '严肃强调'), ('guide_screen_left', '引导屏幕左侧'),
    ('closing_thanks', '总结致谢'),
]


def validate_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value) or '__' in value:
        raise ValueError('ID 仅允许字母、数字、下划线和短横线，最长 64 位；不可包含双下划线')
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('w', encoding='utf-8') as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class PackageManager:
    def __init__(self, data_root='data'):
        self.data = Path(data_root)
        self.root = self.data / 'avatar_packages'
        self.avatars = self.data / 'avatars'
        self.lock = threading.RLock()
        self.restarting = False
        self.recover()

    def recover(self):
        for file in self.root.glob('*/package.json'):
            package = json.loads(file.read_text())
            changed = False
            for action in package['actions']:
                if action['status'] in ('uploading', 'pending', 'running'):
                    action.update(status='failed', error='服务已重启，请重新上传该动作')
                    changed = True
            if changed:
                atomic_json(file, package)

    def _path(self, package_id):
        return self.root / validate_id(package_id) / 'package.json'

    def get(self, package_id):
        with self.lock:
            path = self._path(package_id)
            if not path.is_file():
                raise FileNotFoundError('角色包不存在')
            return json.loads(path.read_text())

    def list(self):
        with self.lock:
            return [self.get(path.parent.name) for path in sorted(self.root.glob('*/package.json'))]

    def _save(self, package):
        package['updated_at'] = time.time()
        atomic_json(self._path(package['package_id']), package)

    def create(self, package_id, name, actions=None):
        validate_id(package_id)
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError('角色名称长度必须为 1–80 个字符')
        actions = actions if actions is not None else [dict(id=i, label=l) for i, l in DEFAULT_ACTIONS]
        if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
            raise ValueError('每个角色包需要 1–8 个动作')
        items, ids = [], set()
        for item in actions:
            if not isinstance(item, dict) or 'id' not in item:
                raise ValueError('每个动作必须包含 ID 和标签')
            action_id = validate_id(item['id'])
            label = item.get('label', action_id)
            if not isinstance(label, str) or not 1 <= len(label.strip()) <= 40 or action_id in ids:
                raise ValueError('动作标签无效或动作 ID 重复')
            ids.add(action_id)
            items.append(dict(id=action_id, label=label.strip(), avatar_id=f'{package_id}__{action_id}',
                              status='empty', progress=0, error='', validation=None))
        if 'neutral_explain' not in ids:
            raise ValueError('角色包必须包含 neutral_explain 动作')
        with self.lock:
            if self.restarting:
                raise ValueError('服务即将重启')
            if self._path(package_id).exists():
                raise ValueError('角色包 ID 已存在，请使用新的 ID')
            package = dict(schema_version=1, package_id=package_id, name=name.strip(), model='wav2lip',
                           img_size=256, fps=25, default_action='neutral_explain', published=False,
                           actions=items, created_at=time.time())
            self._save(package)
            return package

    def action(self, package, action_id):
        validate_id(action_id)
        for action in package['actions']:
            if action['id'] == action_id:
                return action
        raise ValueError('该动作不属于此角色包')

    def reserve_upload(self, package_id, action_id):
        with self.lock:
            package = self.get(package_id)
            action = self.action(package, action_id)
            if self.restarting or package['published'] or action['status'] in ('uploading', 'pending', 'running'):
                raise ValueError('角色包已发布或动作正在处理；已发布角色请创建新版本')
            if action['status'] == 'completed':
                raise ValueError('已生成动作不可覆盖，请创建新版本角色包')
            action.update(status='uploading', progress=0, error='', validation=None)
            self._save(package)

    def update_action(self, package_id, action_id, **values):
        with self.lock:
            package = self.get(package_id)
            self.action(package, action_id).update(**values)
            self._save(package)

    def enqueue(self, package_id, action_id, source):
        from server.task_manager import task_manager
        with self.lock:
            package = self.get(package_id)
            action = self.action(package, action_id)
            target = self.root / package_id / 'source' / f'{action_id}.mp4'
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            task_id = str(uuid.uuid4())
            action.update(status='pending', task_id=task_id, progress=0)
            self._save(package)
            task_manager.add_task('wav2lip', action['avatar_id'], {}, task_id=task_id,
                                  runner=lambda progress: self.generate(package_id, action_id, progress))
            return task_id

    def generate(self, package_id, action_id, task_progress):
        from avatars.wav2lip.genavatar import generate_avatar
        from avatars.wav2lip.face_detection.detection.sfd.sfd_detector import SFDDetector
        import torch
        staging = None
        report = None

        def progress(value):
            task_progress(value)
            self.update_action(package_id, action_id, progress=value)

        try:
            self.update_action(package_id, action_id, status='running')
            package = self.get(package_id)
            action = self.action(package, action_id)
            source = self.root / package_id / 'source' / f'{action_id}.mp4'
            report = probe_video(source, package.get('video_size'))
            weights = Path(os.environ.get('LIVETALKING_SFD_PATH', 'models/s3fd.pth'))
            if not weights.is_file():
                raise VideoValidationError('缺少 S3FD 检测模型，请由管理员准备 models/s3fd.pth', report)
            detector = SFDDetector('cuda' if torch.cuda.is_available() else 'cpu', str(weights))
            boxes = validate_faces(source, report, detector, progress)
            del detector
            self.update_action(package_id, action_id, validation=report)
            self.avatars.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix='.package-', dir=self.avatars))
            generate_avatar(str(source), action['avatar_id'], str(staging), img_size=256,
                            validated_boxes=boxes, progress_callback=lambda p: progress(65 + int(p * .34)))
            destination = self.avatars / action['avatar_id']
            with self.lock:
                if destination.exists():
                    raise ValueError('Avatar 输出目录已存在，拒绝覆盖')
                os.replace(staging / action['avatar_id'], destination)
                package = self.get(package_id)
                package['video_size'] = [report['width'], report['height']]
                self.action(package, action_id).update(status='completed', progress=100, error='', validation=report)
                self._save(package)
            task_progress(100)
        except Exception as exc:
            self.update_action(package_id, action_id, status='failed', error=str(exc),
                               validation=getattr(exc, 'report', report))
            raise
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    def publish(self, package_id, restart=False):
        with self.lock:
            if self.restarting:
                raise ValueError('服务正在准备重启')
            package = self.get(package_id)
            if any(a['status'] != 'completed' for a in package['actions']):
                raise ValueError('全部动作完成后才能发布角色包')
            if any(a['status'] in ('pending', 'running', 'uploading') for p in self.list() for a in p['actions']):
                raise ValueError('仍有上传或生成任务，请完成后发布')
            for action in package['actions']:
                folder = self.avatars / action['avatar_id']
                if not (folder / 'coords.pkl').is_file() or not (folder / 'face_imgs').is_dir() or not (folder / 'full_imgs').is_dir():
                    raise ValueError('动作文件不完整，无法发布')
                expected_count = action.get('validation', {}).get('frame_count')
                if not expected_count or len(list((folder/'face_imgs').glob('*.png'))) != expected_count or len(list((folder/'full_imgs').glob('*.png'))) != expected_count:
                    raise ValueError('动作图片帧数不完整，无法发布')
            package['published'] = True
            package['published_at'] = time.time()
            atomic_json(self.root / package_id / 'manifest.json', package)
            self._save(package)
            atomic_json(self.root / 'active.json', dict(package_id=package_id,
                        avatar_id=self.action(package, 'neutral_explain')['avatar_id']))
            self.restarting = restart
            return package

    def active_avatar(self):
        path = self.root / 'active.json'
        if not path.exists():
            return None
        active = json.loads(path.read_text())
        package = self.get(active['package_id'])
        if not package['published']:
            raise ValueError('活动角色尚未发布')
        return self.action(package, package['default_action'])['avatar_id']

    def preview_allowed(self, avatar_id):
        if '__' not in avatar_id:
            validate_id(avatar_id)
            return
        package_id, action_id = avatar_id.split('__', 1)
        action = self.action(self.get(package_id), action_id)
        if action['status'] != 'completed':
            raise ValueError('动作素材尚未生成完成')


package_manager = PackageManager()
