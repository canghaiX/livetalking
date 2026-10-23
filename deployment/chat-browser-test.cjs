// Live integration test: calls the configured LLM/TTS and may incur API charges.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || '/tmp/livetalking-browser/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');

(async () => {
  const base = process.env.BROWSER_BASE || 'http://127.0.0.1:8020';
  const output = '/tmp/livetalking-chat-live';
  await fs.mkdir(output, {recursive: true});
  const browser = await chromium.launch({headless: true, args: [
    '--no-sandbox', '--autoplay-policy=no-user-gesture-required',
  ]});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', exception => errors.push(exception.message));
    await page.goto(base + '/index.html');
    await page.locator('#connectButton').click();
    await page.waitForFunction(() => pc?.connectionState === 'connected' && video.videoWidth > 0,
                               null, {timeout: 60000});
    await page.evaluate(async () => {
      window.receivedSpeech = [];
      events.addEventListener('message', event => window.receivedSpeech.push(JSON.parse(event.data)));
      window.chatAudio = new AudioContext();
      await window.chatAudio.resume();
      window.chatAnalyser = window.chatAudio.createAnalyser();
      window.chatAudio.createMediaStreamSource(ui.video.srcObject).connect(window.chatAnalyser);
      window.chatPeak = 0;
      window.chatTimer = setInterval(() => {
        const samples = new Uint8Array(window.chatAnalyser.fftSize);
        window.chatAnalyser.getByteTimeDomainData(samples);
        for (const sample of samples) window.chatPeak = Math.max(window.chatPeak, Math.abs(sample - 128));
      }, 30);
      window.chatChunks = [];
      window.chatRecorder = new MediaRecorder(ui.video.srcObject, {mimeType: 'video/webm;codecs=vp8,opus'});
      window.chatRecorder.ondataavailable = event => window.chatChunks.push(event.data);
      window.chatRecorder.start(1000);
    });
    await page.locator('#messageInput').fill('请用五百字详细介绍铁路科研的历史。');
    await page.locator('#messageInput').press('Enter');
    await page.waitForTimeout(200);
    await page.locator('#messageInput').fill('请只用两句简短的话介绍铁科院，然后向来宾致谢告别，总计不超过60字。');
    await page.locator('#messageInput').press('Enter');
    await page.waitForFunction(() => currentRequest && !currentRequest.message.classList.contains('pending'),
                               null, {timeout: 90000});
    const answer = await page.evaluate(() => currentRequest.text.textContent);
    assert.ok(!['回复失败', '暂未收到模型回复'].includes(answer), answer);
    await page.waitForFunction(() => !ui.subtitle.hidden && ui.subtitle.textContent.length > 0,
                               null, {timeout: 60000});
    await page.screenshot({path: output + '/desktop-speaking.png', fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: output + '/mobile-speaking.png', fullPage: true});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.waitForFunction(() => {
      const end = window.receivedSpeech.filter(event => event.status === 'end' && event.request_id === currentRequest.id);
      return end.some(event => event.round_final);
    }, null, {timeout: 90000});
    await page.waitForFunction(() => ui.subtitle.hidden, null, {timeout: 5000});
    await page.waitForTimeout(11000);
    const result = await page.evaluate(async () => {
      const finished = new Promise(resolve => window.chatRecorder.onstop = resolve);
      window.chatRecorder.stop();
      await finished;
      const bytes = new Uint8Array(await new Blob(window.chatChunks).arrayBuffer());
      let binary = '';
      for (let i = 0; i < bytes.length; i += 32768) binary += String.fromCharCode(...bytes.subarray(i, i + 32768));
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 100;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(ui.video, 0, 0, 100, 100);
      const pixels = ctx.getImageData(0, 0, 100, 100).data;
      const pixelSum = pixels.reduce((sum, value, index) => sum + (index % 4 !== 3 ? value : 0), 0);
      const stats = [];
      for (const item of (await pc.getStats()).values()) if (item.type === 'inbound-rtp') {
        stats.push({kind: item.kind, packets: item.packetsReceived, frames: item.framesDecoded});
      }
      return {answer: currentRequest.text.textContent, events: window.receivedSpeech,
              peak: window.chatPeak, pixelSum, stats, recording: btoa(binary)};
    });
    assert.ok(result.peak > 2, 'Expected audible audio in browser MediaStream');
    assert.ok(result.pixelSum > 10000, 'Expected nonblank video');
    assert.ok(result.stats.every(item => item.packets > 0));
    assert.deepEqual(errors, []);
    await fs.writeFile(output + '/conversation.webm', Buffer.from(result.recording, 'base64'));
    delete result.recording;
    await fs.writeFile(output + '/results.json', JSON.stringify({...result, errors}, null, 2));
    console.log(JSON.stringify({...result, output, errors}));
    await page.evaluate(() => { clearInterval(window.chatTimer); window.chatAudio.close(); disconnect(); });
  } finally { await browser.close(); }
})().catch(exception => { console.error(exception); process.exit(1); });
