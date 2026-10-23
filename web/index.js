'use strict';

const ui = Object.fromEntries([
  'video', 'subtitle', 'statusDot', 'statusText', 'connectButton', 'videoPlaceholder',
  'connectionError', 'messages', 'empty', 'dialogError', 'chatForm', 'messageInput',
  'voiceButton', 'voiceHint',
].map(id => [id, document.getElementById(id)]));
let pc = null;
let sessionid = '';
let events = null;
let currentRequest = null;
let voice = null;
let captionTimer = null;
let captionSpeech = '';
let connectAbort = null;
let interruption = Promise.resolve();
let latestGeneration = 0;

function error(element, message = '') {
  element.textContent = message;
  element.hidden = !message;
}

async function api(path, body, signal) {
  const response = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body), signal,
  });
  const result = await response.json();
  if (!response.ok || result.code !== 0) throw new Error(result.msg || '请求失败，请重试');
  return result.data || {};
}

function clearCaption() {
  clearTimeout(captionTimer);
  captionSpeech = '';
  ui.subtitle.hidden = true;
  ui.subtitle.textContent = '';
}

function invalidateRequest() {
  if (currentRequest) {
    currentRequest.abort.abort();
    if (currentRequest.message.classList.contains('pending')) {
      currentRequest.message.classList.remove('pending');
      currentRequest.text.textContent = '已打断';
    }
  }
  currentRequest = null;
  clearCaption();
}

function addMessage(role, text, pending = false) {
  ui.empty.hidden = true;
  const item = document.createElement('div');
  item.className = `message ${role}${pending ? ' pending' : ''}`;
  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? '你' : '讲解员';
  const content = document.createElement('div');
  content.className = 'message-text';
  content.textContent = text;
  item.append(label, content);
  ui.messages.append(item);
  ui.messages.scrollTop = ui.messages.scrollHeight;
  return {message: item, text: content};
}

function connected() {
  return pc && pc.connectionState === 'connected' && sessionid;
}

function requestId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)),
    value => value.toString(16).padStart(2, '0')).join('');
}

function status(text, on = false) {
  ui.statusText.textContent = text;
  ui.statusDot.classList.toggle('on', on);
  ui.connectButton.querySelector('span').textContent = pc ? '断开连接' : '开始连接';
  ui.connectButton.querySelector('i').className = pc ? 'bi bi-stop-fill' : 'bi bi-play-fill';
}

function handleEvent(data) {
  const generation = Number(data.dialog_generation);
  if (Number.isFinite(generation)) {
    if (generation < latestGeneration) return;
    latestGeneration = generation;
  }
  if (data.status === 'interrupt') { clearCaption(); return; }
  const request = currentRequest;
  // A request ID also filters late SSE packets queued before an interruption.
  if (!request || data.request_id !== request.id) return;
  if (data.status === 'start') {
    clearCaption();
    captionSpeech = data.speech_id;
    ui.subtitle.textContent = data.text || '';
    ui.subtitle.hidden = !data.text;
    if (request.message.classList.contains('pending') && !request.spoken.has(data.speech_id)) {
      request.spoken.set(data.speech_id, data.text || '');
      request.text.textContent = [...request.spoken.values()].join('');
    }
  } else if (data.status === 'end' && data.speech_id === captionSpeech) {
    const speech = captionSpeech;
    captionTimer = setTimeout(() => {
      if (currentRequest === request && captionSpeech === speech) clearCaption();
    }, 280);
  }
}

function subscribe(id) {
  events = new EventSource(`/sse?sessionid=${encodeURIComponent(id)}`);
  events.onmessage = event => {
    if (sessionid !== id) return;
    try { handleEvent(JSON.parse(event.data)); } catch (_) { /* Ignore invalid events. */ }
  };
}

async function playVideo() {
  try {
    await ui.video.play();
    error(ui.connectionError);
  } catch (_) {
    error(ui.connectionError, '浏览器暂停了有声播放，请点击数字人画面继续');
  }
}

function disconnect() {
  cancelVoice();
  invalidateRequest();
  connectAbort?.abort();
  connectAbort = null;
  events?.close();
  events = null;
  sessionid = '';
  latestGeneration = 0;
  const previous = pc;
  pc = null;
  previous?.close();
  ui.video.srcObject?.getTracks().forEach(track => track.stop());
  ui.video.srcObject = null;
  ui.videoPlaceholder.hidden = false;
  status('未连接');
}

async function connect() {
  if (pc) { disconnect(); return; }
  error(ui.connectionError);
  const peer = new RTCPeerConnection();
  const controller = new AbortController();
  pc = peer;
  connectAbort = controller;
  status('正在连接…');
  const stream = new MediaStream();
  ui.video.srcObject = stream;
  peer.ontrack = event => {
    if (pc !== peer) return;
    if (!stream.getTracks().some(track => track.id === event.track.id)) stream.addTrack(event.track);
    ui.videoPlaceholder.hidden = true;
    playVideo();
  };
  peer.onconnectionstatechange = () => {
    if (pc !== peer) return;
    if (peer.connectionState === 'connected') status('已连接', true);
    if (['failed', 'closed'].includes(peer.connectionState)) {
      disconnect();
      error(ui.connectionError, '连接已断开，请重新连接');
    }
    if (peer.connectionState === 'disconnected') status('网络中断，正在重连…');
  };
  try {
    peer.addTransceiver('video', {direction: 'recvonly'});
    peer.addTransceiver('audio', {direction: 'recvonly'});
    await peer.setLocalDescription(await peer.createOffer());
    await new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timer);
        peer.removeEventListener('icegatheringstatechange', check);
        controller.signal.removeEventListener('abort', abort);
      };
      const check = () => {
        if (peer.iceGatheringState === 'complete') { cleanup(); resolve(); }
      };
      const abort = () => { cleanup(); reject(new DOMException('Cancelled', 'AbortError')); };
      const timer = setTimeout(() => { cleanup(); reject(new Error('网络候选收集超时')); }, 10000);
      peer.addEventListener('icegatheringstatechange', check);
      controller.signal.addEventListener('abort', abort, {once: true});
      check();
    });
    if (pc !== peer) return;
    const response = await fetch('/offer', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(peer.localDescription), signal: controller.signal,
    });
    const answer = await response.json();
    if (!response.ok || !answer.sdp) throw new Error(answer.msg || answer.error || '数字人连接失败');
    if (pc !== peer) return;
    sessionid = answer.sessionid;
    subscribe(sessionid);
    await peer.setRemoteDescription(answer);
    await playVideo();
  } catch (exception) {
    if (pc !== peer) return;
    disconnect();
    if (exception.name !== 'AbortError') error(ui.connectionError, exception.message);
  }
}

async function sendText(value) {
  const text = value.trim();
  if (!text) return;
  if (!connected()) { error(ui.dialogError, '请先连接数字人'); return; }
  cancelVoice();
  invalidateRequest();
  error(ui.dialogError);
  addMessage('user', text);
  const request = {
    id: requestId(), session: sessionid, abort: new AbortController(),
    spoken: new Map(), ...addMessage('assistant', '正在思考…', true),
  };
  currentRequest = request;
  ui.messageInput.value = '';
  try {
    // A preceding voice-start interruption must finish before this chat starts.
    await interruption;
    if (currentRequest !== request) return;
    const result = await api('/human', {
      type: 'chat', text, interrupt: true, sessionid: request.session, request_id: request.id,
    }, request.abort.signal);
    if (currentRequest !== request || result.cancelled) return;
    request.message.classList.remove('pending');
    request.text.textContent = result.answer || '暂未收到模型回复';
    ui.messages.scrollTop = ui.messages.scrollHeight;
  } catch (exception) {
    if (currentRequest !== request || exception.name === 'AbortError') return;
    request.message.classList.remove('pending');
    request.text.textContent = '回复失败';
    error(ui.dialogError, exception.message);
  }
}

function stopCapture(state) {
  state.source?.disconnect();
  state.node?.disconnect();
  state.gain?.disconnect();
  state.stream?.getTracks().forEach(track => track.stop());
  state.context?.close().catch(() => {});
  state.context = null;
}

function cancelVoice() {
  const state = voice;
  if (!state) return;
  voice = null;
  clearTimeout(state.timer);
  clearTimeout(state.finalTimer);
  state.abort.abort();
  stopCapture(state);
  state.ws?.close();
  ui.voiceButton.classList.remove('recording');
  ui.voiceButton.setAttribute('aria-pressed', 'false');
  ui.voiceHint.textContent = '';
}

function voiceError(state, message) {
  if (voice !== state) return;
  cancelVoice();
  error(ui.dialogError, message);
}

async function startVoice() {
  if (voice) return;
  if (!connected()) { error(ui.dialogError, '请先连接数字人'); return; }
  invalidateRequest();
  error(ui.dialogError);
  const state = {held: true, finishing: false, abort: new AbortController()};
  voice = state;
  ui.voiceButton.classList.add('recording');
  ui.voiceButton.setAttribute('aria-pressed', 'true');
  ui.voiceHint.textContent = '正在连接语音…';
  interruption = api('/interrupt_talk', {sessionid}).catch(() => {});
  try {
    await interruption;
    const response = await fetch('/api/asr/aliyun/status', {signal: state.abort.signal});
    const result = await response.json();
    if (!response.ok || !result.data?.available) throw new Error(result.data?.message || result.msg || 'ASR 未配置');
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('麦克风需要 HTTPS 或 localhost 安全地址，请在安全地址下访问并授权麦克风');
    }
    if (voice !== state || !state.held) return;
    state.stream = await navigator.mediaDevices.getUserMedia({audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true,
    }});
    if (voice !== state || !state.held) { stopCapture(state); return; }
    state.context = new AudioContext();
    await state.context.resume();
    await state.context.audioWorklet.addModule('/audio-capture-worklet.js');
    if (voice !== state || !state.held) { stopCapture(state); return; }
    const url = new URL('/api/asr/aliyun', location.href);
    url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    url.searchParams.set('sessionid', sessionid);
    state.ws = new WebSocket(url);
    state.ws.onmessage = event => {
      if (voice !== state) return;
      const data = JSON.parse(event.data);
      if (data.type === 'ready' && state.held) {
        state.node = new AudioWorkletNode(state.context, 'pcm-capture');
        state.node.port.onmessage = message => {
          if (voice !== state) return;
          if (message.data.type === 'pcm' && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(message.data.data);
          }
          if (message.data.type === 'flushed') state.flushed?.();
        };
        state.source = state.context.createMediaStreamSource(state.stream);
        state.gain = state.context.createGain();
        state.gain.gain.value = 0;
        state.source.connect(state.node).connect(state.gain).connect(state.context.destination);
        state.recording = true;
        ui.voiceHint.textContent = '正在聆听…';
        clearTimeout(state.timer);
        state.timer = setTimeout(finishVoice, 60000);
      } else if (data.type === 'partial') {
        ui.voiceHint.textContent = data.text || (state.finishing ? '识别中…' : '正在聆听…');
      } else if (data.type === 'final' && state.finishing && !state.submitted) {
        state.submitted = true;
        const text = (data.text || '').trim();
        cancelVoice();
        if (text) sendText(text);
        else error(ui.dialogError, '未识别到有效语音，请重试');
      } else if (data.type === 'error') {
        voiceError(state, data.message || '语音识别失败');
      }
    };
    state.ws.onerror = () => voiceError(state, '语音连接失败，请检查阿里云 ASR 配置和网络');
    state.ws.onclose = () => voiceError(state, '语音连接已关闭，请重试');
    state.timer = setTimeout(() => voiceError(state, '语音服务连接超时，请重试'), 15000);
  } catch (exception) {
    if (exception.name !== 'AbortError') voiceError(state,
      ['NotAllowedError', 'PermissionDeniedError'].includes(exception.name)
        ? '麦克风权限被拒绝，请在浏览器地址栏授权麦克风' : exception.message);
  }
}

async function finishVoice() {
  const state = voice;
  if (!state || state.finishing) return;
  state.held = false;
  if (!state.recording) { cancelVoice(); return; }
  state.finishing = true;
  clearTimeout(state.timer);
  ui.voiceButton.classList.remove('recording');
  ui.voiceButton.setAttribute('aria-pressed', 'false');
  ui.voiceHint.textContent = '识别中…';
  await new Promise(resolve => {
    const timeout = setTimeout(resolve, 500);
    state.flushed = () => { clearTimeout(timeout); resolve(); };
    state.node.port.postMessage({type: 'flush'});
  });
  if (voice !== state) return;
  stopCapture(state);
  if (state.ws.readyState !== WebSocket.OPEN) { voiceError(state, '语音连接已断开'); return; }
  state.ws.send(JSON.stringify({action: 'finish-task'}));
  state.finalTimer = setTimeout(() => voiceError(state, '识别结果等待超时，请重试'), 15000);
}

ui.connectButton.addEventListener('click', connect);
ui.video.addEventListener('click', playVideo);
ui.chatForm.addEventListener('submit', event => { event.preventDefault(); sendText(ui.messageInput.value); });
ui.messageInput.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendText(ui.messageInput.value);
  }
});
ui.voiceButton.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  event.preventDefault();
  ui.voiceButton.setPointerCapture(event.pointerId);
  startVoice();
});
ui.voiceButton.addEventListener('pointerup', finishVoice);
ui.voiceButton.addEventListener('pointercancel', cancelVoice);
ui.voiceButton.addEventListener('contextmenu', event => event.preventDefault());
ui.voiceButton.addEventListener('keydown', event => {
  if (event.code === 'Space' && !event.repeat) { event.preventDefault(); startVoice(); }
});
ui.voiceButton.addEventListener('keyup', event => {
  if (event.code === 'Space') { event.preventDefault(); finishVoice(); }
});
window.addEventListener('blur', cancelVoice);
window.addEventListener('pagehide', disconnect);
