const $ = id => document.getElementById(id);
const defaults = [['neutral_explain','中性讲解'],['welcome_smile','欢迎微笑'],['serious_emphasis','严肃强调'],['guide_screen_left','引导屏幕左侧'],['closing_thanks','总结致谢']];
const states = {empty:'待上传',uploading:'上传中',pending:'排队中',running:'校验 / 生成中',completed:'已完成',failed:'失败'};
let selected = null, packages = [], restartSupported = false, detectorReady = false, publishing = false;
let pc = null, sessionid = null, previewToken = 0, refreshing = false;
let restartNotBefore = 0;
const esc = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const api = async (url, options = {}) => {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.code !== 0) throw new Error(data.msg || `HTTP ${response.status}`);
  return data.data;
};
function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
function current() { return packages.find(p => p.package_id === selected); }
function showTab(preview) {
  $('materialsPanel').hidden = preview; $('previewPanel').hidden = !preview;
  $('materialsTab').setAttribute('aria-selected', String(!preview));
  $('previewTab').setAttribute('aria-selected', String(preview));
}
function actionRow(action) {
  const row = document.createElement('div');
  row.className = 'action-row'; row.dataset.id = action.id;
  row.innerHTML = `<div><div class="action-title">${esc(action.label)}</div><div class="action-id">${esc(action.id)}</div><div class="upload-line"><input type="file" accept=".mp4,video/mp4" aria-label="${esc(action.label)}视频"><button class="upload-button">↑ 上传解析</button></div><label class="visual-confirm"><input type="checkbox">同一人物、服装、机位与光照；头顶和双手完整，脸部无遮挡</label><div class="row-buttons"><button class="source-button" hidden>查看原视频</button><button class="preview-button" hidden>▶ 预览口型</button></div></div><div class="action-status"><strong></strong><progress max="100" value="0"></progress><div class="report"></div><div class="error" role="alert"></div></div>`;
  row.querySelector('.upload-button').onclick = () => upload(row, action.id);
  row.querySelector('.source-button').onclick = () => {
    $('sourceTitle').textContent = action.label;
    $('sourceVideo').src = `/api/avatar/package/${encodeURIComponent(selected)}/source/${encodeURIComponent(action.id)}?v=${Date.now()}`;
    $('sourceDialog').showModal();
  };
  row.querySelector('.preview-button').onclick = () => { showTab(true); $('previewAction').value = action.id; connect(); };
  return row;
}
function render() {
  $('packageList').replaceChildren(...packages.map(p => {
    const button = document.createElement('button');
    button.className = p.package_id === selected ? 'selected' : '';
    button.innerHTML = `<span>${esc(p.name)}</span><small>${esc(p.package_id)} · ${p.published ? '已发布' : '草稿'}</small>`;
    button.onclick = () => { if (selected !== p.package_id) { disconnect(); selected = p.package_id; $('actions').replaceChildren(); render(); } };
    return button;
  }));
  $('emptyList').hidden = packages.length > 0;
  const package_ = current();
  $('packageEditor').hidden = !package_; $('emptyEditor').hidden = !!package_;
  if (!package_) return;
  $('packageName').textContent = package_.name; $('packageId').textContent = package_.package_id;
  const completed = package_.actions.filter(a => a.status === 'completed').length;
  $('completion').textContent = `${completed} / ${package_.actions.length} 动作完成${package_.published ? ' · 已发布' : ''}`;
  $('publish').disabled = completed !== package_.actions.length || !restartSupported || publishing || packages.some(p => p.actions.some(a => ['uploading','pending','running'].includes(a.status)));
  $('publish').textContent = package_.published ? '设为默认并重启' : '发布并重启';
  for (const action of package_.actions) {
    let row = [...$('actions').children].find(r => r.dataset.id === action.id);
    if (!row) { row = actionRow(action); $('actions').append(row); }
    const busy = ['uploading','pending','running'].includes(action.status) || row.dataset.uploading === 'true';
    const locked = package_.published || action.status === 'completed' || busy || publishing;
    row.querySelector('input[type=file]').disabled = locked;
    row.querySelector('input[type=checkbox]').disabled = locked;
    row.querySelector('.upload-button').disabled = locked || !detectorReady;
    const state = row.querySelector('strong'); state.textContent = states[action.status] || action.status;
    state.className = `state-${action.status}`;
    const progress = row.querySelector('progress');
    if (row.dataset.uploading !== 'true') progress.value = action.progress;
    progress.hidden = !busy;
    const v = action.validation;
    row.querySelector('.report').textContent = v && Number.isFinite(v.duration) ? `${v.width}×${v.height} · ${v.fps}fps · ${v.duration.toFixed(2)}秒${v.face_coverage !== undefined ? ` · 人脸 ${(v.face_coverage*100).toFixed(1)}%` : ''}${v.repaired_frames?.length ? ` · 相邻检测框修复 ${v.repaired_frames.length} 帧` : ''}` : '';
    row.querySelector('.error').textContent = row.dataset.localError || action.error;
    row.querySelector('.source-button').hidden = !action.task_id;
    row.querySelector('.preview-button').hidden = action.status !== 'completed';
  }
  const previousAction = $('previewAction').value;
  const playable = package_.actions.filter(a => a.status === 'completed');
  $('previewAction').replaceChildren(...playable.map(a => new Option(a.label, a.id)));
  if (playable.some(a => a.id === previousAction)) $('previewAction').value = previousAction;
  $('connect').disabled = !playable.length || publishing;
}
async function refresh() {
  if (refreshing || (publishing && Date.now() < restartNotBefore)) return;
  refreshing = true;
  try {
    const data = await api('/api/avatar/packages');
    packages = data.packages; restartSupported = data.restart_supported; detectorReady = data.detector_ready;
    if (!selected && packages.length) { selected = packages[0].package_id; }
    $('serviceState').textContent = detectorReady ? '服务在线 · 检测模型就绪' : '服务在线 · 缺少检测模型';
    if (publishing) { publishing = false; notice('服务已恢复，角色包已发布。'); }
    render();
  } catch (error) { $('serviceState').textContent = publishing ? '服务正在重启' : '服务连接失败'; }
  finally { refreshing = false; }
}
function draftRow(id = '', label = '') {
  const row = document.createElement('div'); row.className = 'draft-row';
  row.innerHTML = `<input required maxlength="64" aria-label="动作 ID" placeholder="action_id" value="${esc(id)}"><input required maxlength="40" aria-label="动作名称" placeholder="动作名称" value="${esc(label)}"><button type="button" class="icon-button" title="删除动作" aria-label="删除动作">×</button>`;
  row.querySelector('button').disabled = id === 'neutral_explain';
  if (id === 'neutral_explain') row.querySelector('input').readOnly = true;
  row.querySelector('button').onclick = () => row.remove();
  return row;
}
$('newPackage').onclick = () => {
  $('createForm').reset(); $('createError').textContent = '';
  $('draftActions').replaceChildren(...defaults.map(([id,label]) => draftRow(id,label)));
  $('createDialog').showModal();
};
$('closeDialog').onclick = () => $('createDialog').close();
$('addAction').onclick = () => { if ($('draftActions').children.length < 8) $('draftActions').append(draftRow()); };
$('createForm').onsubmit = async event => {
  event.preventDefault(); $('createSubmit').disabled = true;
  const form = new FormData(event.target);
  const actions = [...$('draftActions').children].map(row => ({id:row.children[0].value.trim(),label:row.children[1].value.trim()}));
  try {
    const package_ = await api('/api/avatar/package', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({package_id:form.get('package_id').trim(), name:form.get('name').trim(), actions})});
    selected = package_.package_id; $('actions').replaceChildren(); $('createDialog').close(); showTab(false); await refresh();
  } catch (error) { $('createError').textContent = error.message; }
  finally { $('createSubmit').disabled = false; }
};
async function upload(row, actionId) {
  const fileInput = row.querySelector('input[type=file]'); const file = fileInput.files[0];
  const rowError = row.querySelector('.error'); rowError.textContent = ''; row.dataset.localError = '';
  const fail = message => { row.dataset.localError = message; rowError.textContent = message; };
  if (!file || !file.name.toLowerCase().endsWith('.mp4')) { fail('请选择 MP4 视频'); return; }
  if (file.size > 100*1024*1024) { fail('视频大小不可超过 100MB'); return; }
  if (!row.querySelector('input[type=checkbox]').checked) { fail('请完成素材画面确认'); return; }
  row.dataset.uploading = 'true'; row.querySelector('.upload-button').disabled = true; fileInput.disabled = true;
  const progress = row.querySelector('progress'); progress.hidden = false; progress.value = 0;
  const form = new FormData(); form.append('video_file', file);
  const packageId = selected;
  try {
    await new Promise((resolve,reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `/api/avatar/package/${encodeURIComponent(packageId)}/action?action_id=${encodeURIComponent(actionId)}&visual_confirmed=true`);
      xhr.timeout = 180000;
      xhr.upload.onprogress = event => { if (event.lengthComputable) progress.value = event.loaded / event.total * 100; };
      xhr.onload = () => { try { const data = JSON.parse(xhr.responseText); if (xhr.status >= 400 || data.code !== 0) throw new Error(data.msg || '上传失败'); resolve(); } catch(error) { reject(error); } };
      xhr.onerror = () => reject(new Error('上传连接失败，请重试'));
      xhr.ontimeout = () => reject(new Error('上传超时，请重试'));
      xhr.send(form);
    });
    fileInput.value = '';
  } catch(error) { fail(error.message); }
  finally { row.dataset.uploading = 'false'; await refresh(); }
}
function disconnect() {
  previewToken++; if (pc) pc.close(); pc = null; sessionid = null;
  $('liveVideo').srcObject = null; $('liveAudio').srcObject = null;
  $('disconnect').disabled = true; $('sendAudio').disabled = true;
  $('connectionState').textContent = '未连接'; $('sessionInfo').textContent = '';
}
async function connect() {
  const action = current()?.actions.find(a => a.id === $('previewAction').value);
  if (!action || action.status !== 'completed') return;
  disconnect(); const token = previewToken;
  const connection = new RTCPeerConnection({iceServers:[]}); pc = connection;
  $('connectionState').textContent = '连接中'; $('disconnect').disabled = false;
  connection.ontrack = event => {
    if (token !== previewToken) return;
    const target = event.track.kind === 'video' ? $('liveVideo') : $('liveAudio');
    target.srcObject = new MediaStream([event.track]); target.play().catch(() => notice('浏览器暂停了音频播放，请允许当前站点播放音频。'));
  };
  connection.onconnectionstatechange = () => {
    if (token !== previewToken) return;
    const state = connection.connectionState;
    $('connectionState').textContent = ({connected:'已连接',connecting:'连接中',failed:'连接失败',closed:'已断开',disconnected:'连接中断'})[state] || state;
    $('sendAudio').disabled = state !== 'connected' || !sessionid;
  };
  try {
    connection.addTransceiver('video',{direction:'recvonly'}); connection.addTransceiver('audio',{direction:'recvonly'});
    await connection.setLocalDescription(await connection.createOffer());
    await new Promise((resolve,reject) => {
      if (connection.iceGatheringState === 'complete') { resolve(); return; }
      const timeout = setTimeout(() => { connection.removeEventListener('icegatheringstatechange',check); reject(new Error('ICE 协商超时')); },15000);
      function check() { if (connection.iceGatheringState === 'complete') { clearTimeout(timeout); connection.removeEventListener('icegatheringstatechange',check); resolve(); } }
      connection.addEventListener('icegatheringstatechange',check);
    });
    if (token !== previewToken) return;
    const response = await fetch('/offer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sdp:connection.localDescription.sdp,type:'offer',avatar:action.avatar_id})});
    const answer = await response.json();
    if (!response.ok || !answer.sdp) throw new Error(answer.msg || '无法创建数字人连接');
    if (token !== previewToken) { await connection.setRemoteDescription(answer); connection.close(); return; }
    sessionid = answer.sessionid;
    await connection.setRemoteDescription({type:answer.type,sdp:answer.sdp});
    $('sessionInfo').textContent = `Avatar: ${action.avatar_id} · 会话: ${sessionid}`;
    $('sendAudio').disabled = connection.connectionState !== 'connected';
  } catch(error) { if (token === previewToken) { disconnect(); $('connectionState').textContent = error.message; } }
}
$('materialsTab').onclick = () => showTab(false); $('previewTab').onclick = () => showTab(true);
$('connect').onclick = connect; $('disconnect').onclick = disconnect;
$('previewAction').onchange = () => { if (pc) connect(); };
$('sendAudio').onclick = async () => {
  const file = $('testAudio').files[0]; if (!file || !sessionid) { $('audioStatus').textContent = '请选择测试音频并连接预览'; return; }
  const form = new FormData(); form.append('file',file); form.append('sessionid',sessionid);
  $('sendAudio').disabled = true; $('audioStatus').textContent = '发送中';
  try { await api('/humanaudio',{method:'POST',body:form}); $('audioStatus').textContent = '音频已发送'; }
  catch(error) { $('audioStatus').textContent = error.message; }
  finally { $('sendAudio').disabled = !sessionid || pc?.connectionState !== 'connected'; }
};
$('closeSource').onclick = () => $('sourceDialog').close();
$('sourceDialog').addEventListener('close',() => { $('sourceVideo').pause(); $('sourceVideo').removeAttribute('src'); $('sourceVideo').load(); });
$('publish').onclick = () => { $('publishDescription').textContent = `将“${current().name}”的中性讲解设为默认形象，并重启数字人服务。`; $('publishDialog').showModal(); };
$('cancelPublish').onclick = () => $('publishDialog').close();
$('confirmPublish').onclick = async () => {
  $('confirmPublish').disabled = true;
  try {
    await api(`/api/avatar/package/${encodeURIComponent(selected)}/publish`,{method:'POST'});
    disconnect(); publishing = true; restartNotBefore = Date.now() + 6000; $('publishDialog').close(); notice('角色包已发布，正在重启服务…');
    render(); setTimeout(refresh,6000);
  } catch(error) { notice(error.message); $('publishDialog').close(); }
  finally { $('confirmPublish').disabled = false; }
};
window.addEventListener('beforeunload',disconnect);
refresh(); setInterval(refresh,2500);
