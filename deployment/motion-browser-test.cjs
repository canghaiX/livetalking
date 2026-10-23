const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/tmp/livetalking-browser/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');

(async () => {
  const base = process.env.BROWSER_BASE || 'http://127.0.0.1:8020';
  const output = process.env.MOTION_TEST_OUTPUT || '/tmp/livetalking-motion-browser';
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base + '/index.html');
    await page.locator('#connectButton').click();
    await page.waitForFunction(() => pc?.connectionState === 'connected' && document.querySelector('#video').videoWidth > 0,
                               null, { timeout: 60000 });
    const sid = await page.evaluate(() => String(sessionid));
    await page.evaluate(() => {
      window.motionEvents = [];
      window.motionSSE = new EventSource('/sse?sessionid=' + sessionid);
      window.motionSSE.onmessage = message => window.motionEvents.push(JSON.parse(message.data));
      const stream = new MediaStream([
        ...document.querySelector('#video').srcObject.getVideoTracks(),
        ...document.querySelector('#video').srcObject.getAudioTracks(),
      ]);
      window.motionStream = stream;
      window.motionAudio = new AudioContext();
      window.motionAnalyser = window.motionAudio.createAnalyser();
      window.motionAudio.createMediaStreamSource(stream).connect(window.motionAnalyser);
      window.motionPeak = 0;
      window.motionTimer = setInterval(() => {
        const data = new Uint8Array(window.motionAnalyser.fftSize);
        window.motionAnalyser.getByteTimeDomainData(data);
        for (const value of data) window.motionPeak = Math.max(window.motionPeak, Math.abs(value - 128));
      }, 50);
    });
    const cases = [
      ['plain', '请用大约100字介绍铁科院的日常科研工作，保持平实讲解，不要指向屏幕或介绍重点成果，不要问候和告别。', 'chat'],
      ['important', '请介绍屏幕展示的高铁安全监测重点科研成果。先说“请看屏幕上的重点科研成果”，再用四个完整句子讲解其作用，总共约180字。不要告别。', 'chat'],
      ['gentle07', '请分三个自然段，用约180字通俗说明铁路科研为什么重要，适当配合温和介绍手势，手势之间保留正常讲解。不介绍具体展品、不引导看屏幕，不要问候和告别。', 'chat'],
      ['safety', '请用约80字严肃提醒展厅参观者遵守安全要求，说明设备危险与安全距离，不要问候和告别，不要引导看屏幕。', 'chat'],
      ['thanks', '参观已结束，请用两句简短的话感谢来宾并告别，不要引导看屏幕。', 'chat'],
      ['manual', '欢迎来到铁路科研展厅。这里展示科研工作与技术应用，我们将为您介绍铁路发展的历史。', 'echo'],
    ];
    const selectedCases = process.env.MOTION_TEST_CASES
      ? cases.filter(([name]) => process.env.MOTION_TEST_CASES.split(',').includes(name))
      : cases;
    const results = [];
    for (const [name, text, type] of selectedCases) {
      await page.evaluate(() => {
        window.motionEvents = [];
        window.motionChunks = [];
        window.motionRecorder = new MediaRecorder(window.motionStream, { mimeType: 'video/webm;codecs=vp8,opus' });
        window.motionRecorder.ondataavailable = event => window.motionChunks.push(event.data);
        window.motionRecorder.start(1000);
      });
      const response = await page.evaluate(async ({ text, type }) => {
        return (await fetch('/human', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, type, sessionid: String(sessionid), interrupt: true }),
        })).json();
      }, { text, type });
      assert.equal(response.code, 0, JSON.stringify(response));
      const answer = response.data?.answer || text;
      await page.waitForFunction(expected => {
        return window.motionEvents.filter(event => event.status === 'end').map(event => event.text || '').join('') === expected;
      }, answer, { timeout: 120000 });
      // Single-pass motions may finish after the final syllable.
      await page.waitForTimeout(name === 'thanks' ? 11000 : 3000);
      const data = await page.evaluate(async () => {
        const stopped = new Promise(resolve => window.motionRecorder.onstop = resolve);
        window.motionRecorder.stop();
        await stopped;
        const bytes = new Uint8Array(await new Blob(window.motionChunks, { type: 'video/webm' }).arrayBuffer());
        let binary = '';
        for (let start = 0; start < bytes.length; start += 32768)
          binary += String.fromCharCode(...bytes.subarray(start, start + 32768));
        const canvas = document.createElement('canvas');
        canvas.width = 100; canvas.height = 100;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(document.querySelector('#video'), 0, 0, 100, 100);
        const pixels = ctx.getImageData(0, 0, 100, 100).data;
        let sum = 0;
        for (let i = 0; i < pixels.length; i += 4) sum += pixels[i] + pixels[i + 1] + pixels[i + 2];
        return { recording: btoa(binary), pixelSum: sum, peak: window.motionPeak, events: window.motionEvents };
      });
      assert.ok(data.pixelSum > 10000);
      assert.ok(data.peak > 2, 'Expected audible browser audio');
      await fs.writeFile(`${output}/${name}.webm`, Buffer.from(data.recording, 'base64'));
      await page.screenshot({ path: `${output}/${name}.png`, fullPage: true });
      results.push({ name, answer, pixelSum: data.pixelSum, peak: data.peak,
                     segments: data.events.filter(event => event.status === 'start').map(({ emotion, gesture }) => ({ emotion, gesture })) });
      console.log(JSON.stringify(results[results.length - 1]));
    }
    await page.evaluate(async () => {
      await fetch('/human', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionid: String(sessionid), type: 'echo', interrupt: true, text: '这是用于检查打断恢复的长文本。'.repeat(15) }) });
    });
    await page.waitForTimeout(2500);
    await page.evaluate(() => api('/interrupt_talk', {sessionid}));
    await page.waitForTimeout(2000);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: `${output}/mobile.png`, fullPage: true });
    const stats = await page.evaluate(async () => {
      const reports = [];
      for (const report of (await pc.getStats()).values())
        if (report.type === 'inbound-rtp') reports.push({ kind: report.kind, packetsReceived: report.packetsReceived, framesDecoded: report.framesDecoded });
      return reports;
    });
    assert.deepEqual(errors, []);
    await fs.writeFile(`${output}/results.json`, JSON.stringify({ sid, results, stats, errors }, null, 2));
    console.log(JSON.stringify({ sid, stats, errors, output }));
    await page.evaluate(() => { window.motionSSE.close(); clearInterval(window.motionTimer); disconnect(); });
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
