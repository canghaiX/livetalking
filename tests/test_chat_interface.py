import asyncio
import json
import os
import queue
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web, WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from avatars.base_avatar import BaseAvatar
from llm import _llm_response_deepseek
from server.aliyun_asr import Transcript, asr_status, proxy_asr
from server.routes import human, interrupt_talk


def avatar_stub():
    avatar = BaseAvatar.__new__(BaseAvatar)
    avatar.speech_epoch = avatar.dialog_generation = 0
    avatar.speech_lock = threading.Lock()
    avatar.llm_dialog_lock = threading.Lock()
    avatar.llm_history = []
    avatar.msgqueues = []
    avatar.res_frame_queue = queue.Queue()
    avatar.motion_scheduler = object()
    avatar.motion_scheduler = SimpleNamespace(interrupt=lambda epoch: None)
    avatar.opt = SimpleNamespace(avatar_id='smile')
    avatar.tts = SimpleNamespace(put_msg_txt=lambda text, event: None, flush_talk=lambda: None)
    return avatar


class GenerationTests(unittest.TestCase):
    def test_slow_old_llm_does_not_block_new_or_pollute_history(self):
        avatar = avatar_stub()
        spoken = []
        avatar.tts.put_msg_txt = lambda text, event: spoken.append(text)
        old_started = threading.Event()
        release_old = threading.Event()

        def segments(client, model, history, message):
            if message == 'old':
                old_started.set()
                release_old.wait(3)
            return [{'text': message + ' answer', 'emotion': 'neutral', 'gesture': 'none'}]

        with patch('llm._llm_client'), patch('llm._llm_model'), \
             patch('llm._deepseek_segments', side_effect=segments), \
             patch('server.avatar_package_manager.package_manager.emotion_avatar_id', return_value='smile'), \
             ThreadPoolExecutor(max_workers=2) as executor:
            old = executor.submit(_llm_response_deepseek, 'old', avatar,
                                  {'_speech_epoch': 0, '_dialog_generation': 0})
            self.assertTrue(old_started.wait(2))
            avatar.flush_talk()
            new = executor.submit(_llm_response_deepseek, 'new', avatar,
                                  {'_speech_epoch': 1, '_dialog_generation': 1})
            try:
                self.assertEqual(new.result(timeout=2), 'new answer')
            finally:
                release_old.set()
            self.assertIsNone(old.result(timeout=2))
        self.assertEqual(spoken, ['new answer'])
        self.assertEqual([item['content'] for item in avatar.llm_history], ['new', 'new answer'])

    def test_caption_public_ids_original_text_and_stale_drop(self):
        avatar = avatar_stub()
        output = queue.Queue()
        avatar.add_msgqueue(output)
        avatar.notify({'status': 'start', 'text': '1950年，12306', '_speech_epoch': 0,
                       '_dialog_generation': 0, '_utterance_id': 'speech', '_request_id': 'request'})
        public = json.loads(output.get_nowait())
        self.assertEqual(public['text'], '1950年，12306')
        self.assertEqual(public['speech_id'], 'speech')
        self.assertEqual(public['request_id'], 'request')
        self.assertFalse(any(key.startswith('_') for key in public))
        avatar.flush_talk()
        self.assertEqual(json.loads(output.get_nowait())['status'], 'interrupt')
        avatar.notify({'status': 'start', '_speech_epoch': 0, 'text': 'old'})
        avatar.put_msg_txt('old', {'_speech_epoch': 0})
        self.assertTrue(output.empty())


class ChatRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_chat_cancels_stale_http_reply_and_echo_is_compatible(self):
        avatar = avatar_stub()
        old_started = threading.Event()
        release_old = threading.Event()

        def llm(text, session, event):
            if text == 'old':
                old_started.set()
                release_old.wait(3)
            return text + ' answer'

        app = web.Application()
        app['llm_response'] = llm
        app.router.add_post('/human', human)
        app.router.add_post('/interrupt_talk', interrupt_talk)
        with patch('server.routes.get_session', return_value=avatar):
            async with TestClient(TestServer(app)) as client:
                old = asyncio.create_task(client.post('/human', json={
                    'type': 'chat', 'text': 'old', 'request_id': 'a'}))
                try:
                    for _ in range(100):
                        if old_started.is_set():
                            break
                        await asyncio.sleep(.01)
                    self.assertTrue(old_started.is_set())
                    result = await client.post('/human', json={
                        'type': 'chat', 'text': 'new', 'request_id': 'b'})
                    data = (await result.json())['data']
                    self.assertEqual(data['answer'], 'new answer')
                    self.assertEqual(data['dialog_generation'], 2)
                finally:
                    release_old.set()
                data = (await (await old).json())['data']
                self.assertTrue(data['cancelled'])
                epoch = avatar.speech_epoch
                result = await client.post('/human', json={'type': 'chat', 'text': ' '})
                self.assertNotEqual((await result.json())['code'], 0)
                self.assertEqual(avatar.speech_epoch, epoch)
                await client.post('/human', json={'type': 'echo', 'text': 'hello'})
                self.assertEqual(avatar.speech_epoch, epoch)


class ASRTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_workspace_is_explicit_and_credentials_are_not_returned(self):
        app = web.Application()
        app.router.add_get('/status', asr_status)
        app.router.add_get('/asr', proxy_asr)
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-secret',
                                    'DASHSCOPE_WORKSPACE_ID': ''}, clear=True):
            async with TestClient(TestServer(app)) as client:
                response = await client.get('/status')
                text = await response.text()
                self.assertNotIn('test-secret', text)
                self.assertEqual(json.loads(text)['data']['missing'], ['DASHSCOPE_WORKSPACE_ID'])
                response = await client.get('/asr')
                self.assertEqual(response.status, 503)
                self.assertIn('DASHSCOPE_WORKSPACE_ID', await response.text())

    async def test_waits_for_started_replaces_partial_and_finishes_once(self):
        started = asyncio.Event()
        received = []
        controls = []

        async def cloud_handler(request):
            self.assertEqual(request.headers['Authorization'], 'Bearer test-secret')
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            run = await socket.receive_json()
            controls.append(run)
            await started.wait()
            await socket.send_json({'header': {'event': 'task-started'}})
            async for message in socket:
                if message.type == WSMsgType.BINARY:
                    received.append(message.data)
                    for text in ['中国', '中国铁路']:
                        await socket.send_json({'header': {'event': 'result-generated'},
                            'payload': {'output': {'sentence': {'sentence_id': 0, 'text': text}}}})
                elif message.type == WSMsgType.TEXT:
                    controls.append(json.loads(message.data))
                    await socket.send_json({'header': {'event': 'result-generated'},
                        'payload': {'output': {'sentence': {'sentence_id': 1, 'text': '欢迎您。',
                                                          'sentence_end': True}}}})
                    await socket.send_json({'header': {'event': 'task-finished'}})
            return socket

        cloud_app = web.Application()
        cloud_app.router.add_get('/cloud', cloud_handler)
        async with TestServer(cloud_app) as cloud_server:
            app = web.Application()
            app.router.add_get('/asr', proxy_asr)
            env = {'DASHSCOPE_API_KEY': 'test-secret', 'DASHSCOPE_WORKSPACE_ID': 'test-space',
                   'DASHSCOPE_ASR_URL': str(cloud_server.make_url('/cloud'))}
            with patch.dict(os.environ, env, clear=True):
                async with TestClient(TestServer(app)) as client:
                    async with client.ws_connect('/asr') as browser:
                        await browser.send_bytes(b'\x00\x01' * 1600)
                        await asyncio.sleep(.03)
                        self.assertEqual(received, [])
                        started.set()
                        self.assertEqual((await browser.receive_json())['type'], 'ready')
                        self.assertEqual((await browser.receive_json())['text'], '中国')
                        self.assertEqual((await browser.receive_json())['text'], '中国铁路')
                        await browser.send_json({'action': 'finish-task'})
                        self.assertEqual((await browser.receive_json())['text'], '中国铁路欢迎您。')
                        self.assertEqual(await browser.receive_json(),
                                         {'type': 'final', 'text': '中国铁路欢迎您。'})
                        self.assertEqual((await browser.receive()).type, WSMsgType.CLOSE)
        self.assertEqual(len(received), 1)
        self.assertEqual(controls[0]['header']['task_id'], controls[1]['header']['task_id'])
        self.assertEqual(controls[1]['header']['action'], 'finish-task')
        self.assertEqual(controls[0]['payload']['parameters']['sample_rate'], 16000)

    async def test_transcript_sentence_revisions(self):
        transcript = Transcript()
        transcript.update({'begin_time': 0, 'text': '第一'})
        transcript.update({'begin_time': 0, 'text': '第一句。', 'sentence_end': True})
        self.assertEqual(transcript.update({'begin_time': 1000, 'text': '第二句。'}), '第一句。第二句。')


if __name__ == '__main__':
    unittest.main()
