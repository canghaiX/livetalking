// Run with PLAYWRIGHT_MODULE pointing to the installed playwright package.
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const vm = require('node:vm');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || '/tmp/livetalking-browser/node_modules/playwright');

async function resamplingTest() {
  const source = await fs.readFile(path.join(__dirname, '../web/audio-capture-worklet.js'), 'utf8');
  for (const rate of [44100, 48000, 16000]) {
    const messages = [];
    let Processor;
    const context = vm.createContext({
      AudioWorkletProcessor: class { constructor() { this.port = {postMessage: value => messages.push(value)}; } },
      sampleRate: rate, registerProcessor: (name, value) => { Processor = value; },
    });
    vm.runInContext(source, context);
    const processor = new Processor();
    for (let offset = 0; offset < rate; offset += 128) {
      processor.process([[new Float32Array(Math.min(128, rate - offset)).fill(.5)]]);
    }
    processor.port.onmessage({data: {type: 'flush'}});
    const chunks = messages.filter(message => message.type === 'pcm').map(message => new Int16Array(message.data));
    assert.equal(chunks.reduce((sum, chunk) => sum + chunk.length, 0), 16000);
    assert.ok(chunks.every(chunk => chunk.every(value => Math.abs(value - 16384) <= 1)));
    assert.equal(messages.at(-1).type, 'flushed');
  }
}

(async () => {
  await resamplingTest();
  const root = path.resolve(__dirname, '../web');
  const server = http.createServer(async (request, response) => {
    try {
      const file = path.join(root, new URL(request.url, 'http://localhost').pathname);
      if (!file.startsWith(root + path.sep)) throw new Error();
      response.setHeader('Content-Type', file.endsWith('.js') ? 'application/javascript' : 'text/html');
      response.end(await fs.readFile(file));
    } catch (_) { response.writeHead(404); response.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({headless: true, args: [
    '--no-sandbox', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
    '--autoplay-policy=no-user-gesture-required',
  ]});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', exception => errors.push(exception.message));
    await page.addInitScript(() => {
      window.testRequests = [];
      window.testSockets = [];
      window.testEvents = [];
      window.RTCPeerConnection = class extends EventTarget {
        constructor() { super(); this.connectionState = 'new'; this.iceGatheringState = 'complete'; }
        addTransceiver() {}
        createOffer() { return Promise.resolve({type: 'offer', sdp: 'test'}); }
        setLocalDescription(value) { this.localDescription = value; return Promise.resolve(); }
        setRemoteDescription() { this.connectionState = 'connected'; this.onconnectionstatechange(); return Promise.resolve(); }
        close() { this.connectionState = 'closed'; this.onconnectionstatechange(); }
      };
      HTMLMediaElement.prototype.play = () => Promise.resolve();
      window.EventSource = class {
        constructor() { window.testEvents.push(this); }
        close() {}
      };
      const realFetch = window.fetch;
      window.fetch = async (url, options) => {
        const body = options?.body && JSON.parse(options.body);
        if (url === '/offer') return new Response(JSON.stringify({type: 'answer', sdp: 'test', sessionid: 'test'}));
        if (url === '/interrupt_talk') return new Response(JSON.stringify({code: 0}));
        if (url === '/human') {
          window.testRequests.push(body);
          if (body.text === '旧问题') await new Promise(resolve => { window.resolveOld = resolve; });
          return new Response(JSON.stringify({code: 0, data: {answer: body.text + '的回答', request_id: body.request_id}}));
        }
        if (url === '/api/asr/aliyun/status') return new Response(JSON.stringify({code: 0, data: {
          available: !window.missingASR, message: '请配置 DASHSCOPE_WORKSPACE_ID',
        }}));
        return realFetch(url, options);
      };
      window.WebSocket = class {
        static OPEN = 1;
        constructor() {
          this.readyState = 1; this.binary = 0; window.testSockets.push(this);
          setTimeout(() => this.onmessage?.({data: JSON.stringify({type: 'ready'})}), 40);
        }
        send(data) {
          if (typeof data !== 'string') { this.binary += data.byteLength; return; }
          if (JSON.parse(data).action === 'finish-task') {
            this.finished = true;
            for (const value of [{type: 'partial', text: '语音'}, {type: 'final', text: '语音问题'}, {type: 'final', text: '语音问题'}]) {
              setTimeout(() => this.onmessage?.({data: JSON.stringify(value)}), 20);
            }
          }
        }
        close() { this.readyState = 3; setTimeout(() => this.onclose?.(), 0); }
      };
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/index.html`);
    await page.locator('#connectButton').click();
    await page.waitForFunction(() => document.getElementById('statusText').textContent === '已连接');
    await page.locator('#messageInput').fill('旧问题');
    await page.locator('#messageInput').press('Enter');
    await page.waitForFunction(() => window.testRequests.length === 1);
    await page.locator('#messageInput').fill('新问题');
    await page.locator('#messageInput').press('Enter');
    await page.waitForFunction(() => window.testRequests.length === 2);
    await page.evaluate(() => window.resolveOld());
    await page.waitForFunction(() => document.getElementById('messages').textContent.includes('新问题的回答'));
    assert.equal(await page.getByText('旧问题的回答', {exact: true}).count(), 0);
    const requests = await page.evaluate(() => window.testRequests);
    assert.ok(requests.every(request => request.type === 'chat' && request.interrupt === true));
    await page.evaluate(({old, current}) => {
      const source = window.testEvents.at(-1);
      source.onmessage({data: JSON.stringify({status: 'start', text: '过期字幕', request_id: old, speech_id: 'old'})});
      source.onmessage({data: JSON.stringify({status: 'start', text: '1950年，12306', request_id: current, speech_id: 'a'})});
    }, {old: requests[0].request_id, current: requests[1].request_id});
    assert.equal(await page.locator('#subtitle').textContent(), '1950年，12306');
    await page.evaluate(id => {
      const source = window.testEvents.at(-1);
      source.onmessage({data: JSON.stringify({status: 'end', request_id: id, speech_id: 'a'})});
      source.onmessage({data: JSON.stringify({status: 'start', text: '下一段字幕', request_id: id, speech_id: 'b'})});
    }, requests[1].request_id);
    await page.waitForTimeout(350);
    assert.equal(await page.locator('#subtitle').textContent(), '下一段字幕');
    await page.evaluate(() => {
      window.testEvents.at(-1).onmessage({data: JSON.stringify({status: 'interrupt', dialog_generation: 3})});
    });
    assert.equal(await page.locator('#subtitle').isHidden(), true);

    const button = page.locator('#voiceButton');
    await button.dispatchEvent('pointerdown', {button: 0, pointerId: 1});
    await button.dispatchEvent('pointerup', {button: 0, pointerId: 1});
    await page.waitForTimeout(100);
    assert.equal(await page.evaluate(() => voice), null);
    await button.focus();
    await page.keyboard.down('Space');
    await page.waitForFunction(() => voice?.recording, null, {timeout: 15000});
    await page.waitForTimeout(350);
    await page.keyboard.up('Space');
    await page.waitForFunction(() => window.testRequests.some(item => item.text === '语音问题'));
    assert.equal(await page.evaluate(() => window.testRequests.filter(item => item.text === '语音问题').length), 1);
    assert.ok(await page.evaluate(() => window.testSockets.at(-1).binary > 0));
    assert.equal(await page.evaluate(() => voice), null);

    await page.keyboard.down('Space');
    await page.waitForFunction(() => voice?.recording);
    await page.locator('#messageInput').fill('文字打断录音');
    await page.locator('#messageInput').press('Enter');
    await page.keyboard.up('Space');
    await page.waitForFunction(() => window.testRequests.at(-1).text === '文字打断录音');
    assert.equal(await page.evaluate(() => voice), null);
    await page.evaluate(() => { window.missingASR = true; });
    await button.focus();
    await page.keyboard.down('Space');
    await page.waitForFunction(() => document.getElementById('dialogError').textContent.includes('DASHSCOPE_WORKSPACE_ID'));
    await page.keyboard.up('Space');
    for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}]) {
      await page.setViewportSize(viewport);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      if (viewport.width > 800) {
        assert.ok(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight));
      }
      await page.screenshot({path: `/tmp/livetalking-chat-${viewport.width}.png`, fullPage: true});
    }
    assert.deepEqual(errors, []);
    console.log('PASS: resampling, chat interruption, stale captions, hold/release, quick cancel, text cancels recording, missing ASR, desktop/mobile');
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(exception => { console.error(exception); process.exit(1); });
