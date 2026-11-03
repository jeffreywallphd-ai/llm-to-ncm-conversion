'use strict';
const $ = s => document.querySelector(s);
let history = [], status = null, busy = false;
const messages = $('#messages'), prompt = $('#prompt'), error = $('#error');
function showError(text) { error.textContent = text; error.hidden = !text; }
function bubble(role, text, meta = '', preview = false) {
  const row = document.createElement('article'); row.className = `message ${role}${preview ? ' preview-response' : ''}`;
  const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = role === 'user' ? 'Y' : 'N';
  const body = document.createElement('div'); body.className = 'message-body';
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? 'You' : (preview ? 'Interface preview' : 'NCM');
  const content = document.createElement('div'); content.className = 'message-text'; content.textContent = text;
  const detail = document.createElement('div'); detail.className = 'message-meta'; detail.textContent = meta;
  body.append(label, content, detail); row.append(avatar, body); messages.append(row);
  $('.conversation').scrollTop = $('.conversation').scrollHeight;
  return row;
}
async function refresh() {
  const response = await fetch('/api/status', {cache: 'no-store'}); status = await response.json();
  if (!response.ok) throw new Error(status.error || 'The local server is unavailable.');
  const connected = status.mode === 'model';
  $('#model-name').textContent = connected ? status.model.label : 'No model attached';
  $('#model-description').textContent = connected ? 'Your trained export is loaded locally and ready for a conversation.' : 'Explore the interface. Connect a trained export when you\'re ready.';
  $('#sidebar-status').replaceChildren();
  $('#sidebar-status').append(document.createElement('i'), document.createTextNode(connected ? ' Ready to chat' : ' Interface preview'));
  $('#mode-badge').classList.toggle('connected', connected);
  $('#mode-badge').replaceChildren(document.createElement('i'), document.createTextNode(connected ? ' Model connected' : ' Preview mode'));
  $('#preview-note').hidden = connected;
  $('#composer-model').textContent = connected ? 'NCM · Parallel generation · Local' : 'NCM playground · Preview';
  $('#request-count').textContent = `${status.requests} / ${status.max_requests} session requests`;
}
$('#chat-form').addEventListener('submit', async event => {
  event.preventDefault(); if (busy || !prompt.value.trim()) return;
  if (!status) { showError('Wait for the local server to connect, then try again.'); return; }
  const text = prompt.value.trim(); showError(''); busy = true; $('#send').disabled = true;
  $('#welcome').hidden = true; bubble('user', text); prompt.value = '';
  const pending = bubble('assistant', status.mode === 'preview' ? 'Opening the preview…' : 'Generating a response…'); pending.classList.add('thinking');
  try {
    const response = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-NCM-Token': status.token}, body: JSON.stringify({messages: [...history, {role: 'user', content: text}]})});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'The request could not complete.');
    pending.remove();
    const preview = result.mode === 'preview';
    const meta = preview ? 'No model attached · No inference performed' : `${result.returned_tokens} tokens · ${result.generation_seconds.toFixed(2)} s · ${result.eos_found ? 'EOS found' : 'Canvas limit reached'}`;
    bubble('assistant', result.text || '(The model returned EOS without visible text.)', meta, preview);
    if (!preview && result.text.trim()) history.push({role: 'user', content: text}, {role: 'assistant', content: result.text});
    await refresh();
  } catch (err) { pending.remove(); showError(err.message); }
  finally { busy = false; $('#send').disabled = false; prompt.focus(); }
});
prompt.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('#chat-form').requestSubmit(); } });
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { prompt.value = button.dataset.prompt; prompt.focus(); }));
document.querySelectorAll('.new-chat').forEach(button => button.addEventListener('click', () => { if (busy) return; history = []; messages.replaceChildren(); $('#welcome').hidden = false; prompt.value = ''; showError(''); prompt.focus(); }));
[$('#connect-button'), $('#mode-badge')].forEach(button => button.addEventListener('click', () => $('#connect-dialog').showModal()));
document.querySelectorAll('.close-dialog').forEach(button => button.addEventListener('click', () => $('#connect-dialog').close()));
refresh().catch(err => showError(err.message));
