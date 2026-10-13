"""End-to-end verification against the isolated port-8021 test container."""
import json
import os
import subprocess
import time
from pathlib import Path

import requests


BASE = 'http://127.0.0.1:8021'
CONTAINER = os.environ.get('VERIFY_CONTAINER', 'livetalking-package-verify')
PACKAGE = 'integration_presenter'


def api(path, method='get', **kwargs):
    response = getattr(requests, method)(BASE + path, timeout=30, **kwargs)
    response.raise_for_status()
    body = response.json()
    assert body['code'] == 0, body
    return body['data']


if os.environ.get('VERIFY_EXISTING'):
    package = api(f'/api/avatar/package/{PACKAGE}')
else:
    package = api('/api/avatar/package', 'post', json={'package_id':PACKAGE,'name':'五动作集成验证'})
for action in package['actions']:
    if action['status'] == 'completed':
        continue
    with (Path(__file__).parent/'test-valid.mp4').open('rb') as video:
        result = api(f"/api/avatar/package/{PACKAGE}/action?action_id={action['id']}&visual_confirmed=true", 'post',
                     files={'video_file':('clip.mp4',video,'video/mp4')})
    print('queued',action['id'],result,flush=True)
deadline = time.monotonic() + 240
while time.monotonic() < deadline:
    package = api(f'/api/avatar/package/{PACKAGE}')
    statuses = [(a['id'],a['status'],a['progress']) for a in package['actions']]
    print(statuses,flush=True)
    failed = [a for a in package['actions'] if a['status'] == 'failed']
    assert not failed, failed
    if all(a['status'] == 'completed' for a in package['actions']):
        break
    time.sleep(3)
else:
    raise AssertionError('Generation did not complete')
for action in package['actions']:
    folder = Path(os.environ['VERIFY_DATA']) / 'avatars' / action['avatar_id']
    assert len(list((folder/'full_imgs').glob('*.png'))) == 75
    assert len(list((folder/'face_imgs').glob('*.png'))) == 75
    assert (folder/'coords.pkl').is_file()
    result = subprocess.run(['docker','exec','-e',f'SMOKE_BASE_URL={BASE}','-e',f"SMOKE_AVATAR={action['avatar_id']}",
                             CONTAINER,'python','/verification/webrtc-smoke-test.py'],
                            capture_output=True,text=True,timeout=75)
    print('preview',action['id'],result.stdout,flush=True)
    assert result.returncode == 0, result.stderr
    time.sleep(.5)
published = api(f'/api/avatar/package/{PACKAGE}/publish','post')
assert published['restarting']
time.sleep(7)
for _ in range(30):
    try:
        config = api('/api/admin/config')['config']
        break
    except (requests.RequestException,ValueError):
        time.sleep(1)
else:
    raise AssertionError('Service did not recover after publication')
assert config['avatar_id'] == PACKAGE + '__neutral_explain', config
package = api(f'/api/avatar/package/{PACKAGE}')
assert package['published'] and all(a['status']=='completed' for a in package['actions'])
print(json.dumps({'five_actions':'completed','five_webrtc_audio_tests':'passed',
                  'default_after_restart':config['avatar_id'],'published':package['published']},ensure_ascii=False),flush=True)
