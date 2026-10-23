"""Bounded, credential-hiding Paraformer realtime WebSocket bridge."""

import asyncio
import json
import os
import re
import uuid

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

MAX_AUDIO_BYTES = 16000 * 2 * 60


def settings():
    key = os.getenv('DASHSCOPE_API_KEY', '').strip()
    workspace = os.getenv('DASHSCOPE_WORKSPACE_ID', '').strip()
    url = os.getenv('DASHSCOPE_ASR_URL', '').strip()
    if workspace and not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', workspace):
        raise ValueError('DASHSCOPE_WORKSPACE_ID 格式无效')
    if not url and workspace:
        url = f'wss://{workspace}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference'
    silence = int(os.getenv('DASHSCOPE_ASR_MAX_SENTENCE_SILENCE', '800'))
    if not 200 <= silence <= 6000:
        raise ValueError('ASR 断句静音时间必须为 200–6000ms')
    missing = [name for name, value in (
        ('DASHSCOPE_API_KEY', key), ('DASHSCOPE_WORKSPACE_ID', workspace)) if not value]
    return key, workspace, url, silence, missing


async def asr_status(request):
    try:
        *_, missing = settings()
        return web.json_response({'code': 0, 'data': {
            'available': not missing, 'missing': missing,
            'message': ('请配置 ' + '、'.join(missing)) if missing else '',
        }})
    except ValueError as exc:
        return web.json_response({'code': -1, 'msg': str(exc)}, status=503)


class Transcript:
    """Replace a sentence's partial results instead of appending revisions."""

    def __init__(self):
        self.sentences = {}

    def update(self, sentence):
        identifier = str(sentence.get('sentence_id', sentence.get('begin_time', 0)))
        text = sentence.get('text', '')
        if isinstance(text, str):
            self.sentences[identifier] = text
        return ''.join(self.sentences.values())


async def proxy_asr(request):
    try:
        key, workspace, url, silence, missing = settings()
    except ValueError as exc:
        return web.json_response({'code': -1, 'msg': str(exc)}, status=503)
    if missing:
        return web.json_response({'code': 'ASR_NOT_CONFIGURED',
                                  'msg': '请配置 ' + '、'.join(missing)}, status=503)

    browser = web.WebSocketResponse(heartbeat=20, max_msg_size=65536)
    await browser.prepare(request)
    task_id = uuid.uuid4().hex
    tasks = []

    async def fail(message):
        if not browser.closed:
            await browser.send_json({'type': 'error', 'message': message})

    try:
        # trust_env shares the configured outbound proxy with other cloud APIs.
        async with ClientSession(trust_env=True, timeout=ClientTimeout(total=None, sock_connect=10)) as session:
            async with session.ws_connect(url, headers={
                'Authorization': f'Bearer {key}', 'X-DashScope-WorkSpace': workspace,
            }, heartbeat=20, max_msg_size=1024 * 1024) as cloud:
                await cloud.send_json({
                    'header': {'action': 'run-task', 'task_id': task_id, 'streaming': 'duplex'},
                    'payload': {'task_group': 'audio', 'task': 'asr', 'function': 'recognition',
                                'model': os.getenv('DASHSCOPE_ASR_MODEL', 'paraformer-realtime-v2'),
                                'parameters': {'format': 'pcm', 'sample_rate': 16000,
                                               'max_sentence_silence': silence}, 'input': {}},
                })
                # Browser capture starts only after this acknowledgement. Early
                # browser packets remain in aiohttp's bounded receive buffer.
                first = await asyncio.wait_for(cloud.receive_json(), 10)
                if first.get('header', {}).get('event') != 'task-started':
                    await fail('阿里云 ASR 未能启动，请检查模型权限和业务空间配置')
                    return browser
                await browser.send_json({'type': 'ready'})
                transcript = Transcript()

                async def from_cloud():
                    async for message in cloud:
                        if message.type != WSMsgType.TEXT:
                            break
                        data = json.loads(message.data)
                        event = data.get('header', {}).get('event')
                        if event == 'result-generated':
                            sentence = data.get('payload', {}).get('output', {}).get('sentence', {})
                            await browser.send_json({'type': 'partial', 'text': transcript.update(sentence)})
                        elif event == 'task-finished':
                            await browser.send_json({'type': 'final', 'text': ''.join(transcript.sentences.values())})
                            return
                        elif event == 'task-failed':
                            await fail('阿里云 ASR 识别失败，请检查模型权限、余额和配置')
                            return
                    raise ConnectionError('ASR connection closed before task-finished')

                async def from_browser():
                    byte_count = 0
                    finished = False
                    async for message in browser:
                        if message.type == WSMsgType.BINARY and not finished:
                            byte_count += len(message.data)
                            if byte_count > MAX_AUDIO_BYTES or len(message.data) % 2:
                                await fail('录音超过 60 秒或 PCM 数据格式无效')
                                return
                            await cloud.send_bytes(message.data)
                        elif message.type == WSMsgType.TEXT:
                            control = json.loads(message.data)
                            if control.get('action') == 'finish-task' and not finished:
                                finished = True
                                await cloud.send_json({
                                    'header': {'action': 'finish-task', 'task_id': task_id,
                                               'streaming': 'duplex'}, 'payload': {'input': {}},
                                })
                        elif message.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                            return

                tasks = [asyncio.create_task(from_cloud()), asyncio.create_task(from_browser())]
                done, _ = await asyncio.wait(tasks, timeout=80, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    await fail('语音识别超时，请重试')
                for task in done:
                    task.result()
    except (TimeoutError, asyncio.TimeoutError):
        await fail('阿里云 ASR 连接或识别超时，请重试')
    except Exception:
        # Do not relay exception strings which may include provider request headers.
        await fail('阿里云 ASR 连接失败，请检查网络、密钥和业务空间配置')
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await browser.close()
    return browser
