import asyncio
import os
import tempfile
from pathlib import Path

from aiohttp import web

from server.avatar_package_manager import package_manager, validate_id
from server.avatar_video_validation import MAX_UPLOAD_BYTES


def ok(data):
    return web.json_response({'code': 0, 'msg': 'ok', 'data': data})


def error(exc, status=400):
    return web.json_response({'code': -1, 'msg': str(exc)}, status=status)


async def create_package(request):
    try:
        params = await request.json()
        return ok(package_manager.create(params['package_id'], params['name'], params.get('actions')))
    except (ValueError, KeyError, TypeError) as exc:
        return error(exc)


async def list_packages(request):
    return ok({'packages': package_manager.list(), 'restart_supported': os.environ.get('LIVETALKING_ALLOW_RESTART') == '1',
               'detector_ready': Path(os.environ.get('LIVETALKING_SFD_PATH', 'models/s3fd.pth')).is_file()})


async def get_package(request):
    try:
        return ok(package_manager.get(request.match_info['package_id']))
    except FileNotFoundError as exc:
        return error(exc, 404)
    except ValueError as exc:
        return error(exc)


async def upload_action(request):
    package_id = request.match_info['package_id']
    action_id = request.query.get('action_id', '')
    temp = None
    reserved = False
    try:
        validate_id(package_id)
        validate_id(action_id)
        if request.query.get('visual_confirmed') != 'true':
            raise ValueError('请确认人物、服装、构图一致，且头手完整、无遮挡')
        if request.content_type != 'multipart/form-data':
            raise ValueError('请使用 multipart/form-data 上传视频')
        package_manager.reserve_upload(package_id, action_id)
        reserved = True
        folder = package_manager.root / package_id / 'source'
        folder.mkdir(parents=True, exist_ok=True)
        reader = await request.multipart()
        uploaded = False
        total = 0
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name != 'video_file' or uploaded or not part.filename or Path(part.filename).suffix.lower() != '.mp4':
                raise ValueError('请求必须且只能包含一个名为 video_file 的 MP4 文件')
            fd, name = tempfile.mkstemp(suffix='.mp4', prefix='.upload-', dir=folder)
            temp = Path(name)
            with os.fdopen(fd, 'wb') as output:
                while True:
                    chunk = await part.read_chunk(256 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError('视频大小不可超过 100MB')
                    output.write(chunk)
            uploaded = True
        if not uploaded or total == 0:
            raise ValueError('未上传视频')
        task_id = package_manager.enqueue(package_id, action_id, temp)
        return ok({'task_id': task_id, 'package_id': package_id, 'action_id': action_id})
    except asyncio.CancelledError:
        if reserved:
            package_manager.update_action(package_id, action_id, status='failed', error='上传已取消，请重试')
        raise
    except Exception as exc:
        if reserved:
            package_manager.update_action(package_id, action_id, status='failed', error=str(exc))
        return error(exc, 404 if isinstance(exc, FileNotFoundError) else 400)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


async def source_video(request):
    try:
        package = package_manager.get(request.match_info['package_id'])
        action = package_manager.action(package, request.match_info['action_id'])
        path = package_manager.root / package['package_id'] / 'source' / f"{action['id']}.mp4"
        if not path.is_file():
            raise FileNotFoundError('尚未上传该动作视频')
        return web.FileResponse(path)
    except (ValueError, FileNotFoundError) as exc:
        return error(exc, 404)


async def publish_package(request):
    from server.task_manager import task_manager
    try:
        if os.environ.get('LIVETALKING_ALLOW_RESTART') != '1':
            raise ValueError('此运行环境未开启受控重启，请在 Docker 部署中发布')
        if any(t['status'] in ('pending', 'running') for t in task_manager.list_tasks()):
            raise ValueError('生成队列尚未完成，请稍后发布')
        package = package_manager.publish(request.match_info['package_id'], restart=True)
        # Exit the container's main process after the HTTP response; Docker restarts it.
        asyncio.get_running_loop().call_later(2, os._exit, 0)
        return ok({'package': package, 'restarting': True})
    except (ValueError, FileNotFoundError) as exc:
        return error(exc)


def setup_package_routes(app):
    app.router.add_post('/api/avatar/package', create_package)
    app.router.add_get('/api/avatar/packages', list_packages)
    app.router.add_get('/api/avatar/package/{package_id}', get_package)
    app.router.add_post('/api/avatar/package/{package_id}/action', upload_action)
    app.router.add_get('/api/avatar/package/{package_id}/source/{action_id}', source_video)
    app.router.add_post('/api/avatar/package/{package_id}/publish', publish_package)
