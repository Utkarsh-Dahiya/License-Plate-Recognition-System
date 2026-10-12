// License Plate Transcription Reviewer logic.
// Runs after window.PLATE_DATA is defined by transcribe_data.js.
//
// GROUND-TRUTH SAFETY RULES (enforced here):
//  * The OCR "hint" is shown for speed only and is NEVER copied into the
//    ground-truth field automatically.
//  * Nothing is auto-verified. Only an explicit human "Verify"/"Save" action
//    records a transcription, and every value is preserved exactly as typed.
//  * Pre-verified entries (V*) are locked read-only until the reviewer clicks
//    "Modify" and confirms; their values round-trip unchanged otherwise.
(function () {
'use strict';

var DATA = window.PLATE_DATA;
if (!DATA) {
  document.body.innerHTML =
    '<pre style="color:#e0566b;padding:20px;font:14px monospace">' +
    'transcribe_data.js failed to load.\n' +
    'Regenerate it with:  python diagnostics/ocr_benchmark/gen_data_js.py</pre>';
  return;
}

var TOTAL = DATA.total;
var IDS = DATA.ids;
var ENTRIES = DATA.entries;
var DRAFT = DATA.draft || {};
var STORAGE_KEY = 'plate_transcribe_draft_v1';

var currentIdx = 0;
var filter = 'pending';               // default: review pending entries first
var view = { zoom: 1, bri: 100, con: 100, gray: false };
var unlocked = {};                     // per-index: verified entry unlocked for edit

var $ = function (id) { return document.getElementById(id); };

var stage = $('stage');
var img = $('img');
var missing = $('missing');
var gt = $('gt');
var warn = $('warn');
var statusLine = $('status');
var saveMsg = $('saveMsg');
var mStatus = $('mStatus');

function blank(i) { return { text: '', unreadable: false, verified: false, review: false }; }
function draftOf(i) { if (!DRAFT[i]) DRAFT[i] = blank(i); return DRAFT[i]; }

function persist() {
  var payload = {
    version: 1,
    currentIdx: currentIdx,
    filter: filter,
    view: { zoom: view.zoom, bri: view.bri, con: view.con, gray: view.gray },
    draft: DRAFT
  };
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(payload)); } catch (e) { /* private mode */ }
}

function restore() {
  var raw;
  try { raw = localStorage.getItem(STORAGE_KEY); } catch (e) { return; }
  if (!raw) return;
  try {
    var s = JSON.parse(raw);
    if (s && s.version === 1) {
      if (typeof s.currentIdx === 'number') currentIdx = s.currentIdx;
      if (typeof s.filter === 'string') filter = s.filter;
      if (s.view) {
        if (typeof s.view.zoom === 'number') view.zoom = s.view.zoom;
        if (typeof s.view.bri === 'number') view.bri = s.view.bri;
        if (typeof s.view.con === 'number') view.con = s.view.con;
        if (typeof s.view.gray === 'boolean') view.gray = s.view.gray;
      }
      if (s.draft && typeof s.draft === 'object') DRAFT = s.draft;
    }
  } catch (e) { /* ignore corrupt storage */ }
}

function showMsg(msg, type) {
  saveMsg.textContent = msg || '';
  saveMsg.className = 'tag' + (type ? ' ' + type : '');
  statusLine.textContent = msg || '';
  statusLine.className = 'statusline' + (type ? ' ' + type : '');
}

function computeStatus(i) {
  var e = ENTRIES[i];
  var d = DRAFT[i] || blank(i);
  if (e.verified) return 'verified';       // dataset-verified (V*) takes precedence
  if (d.verified) return 'verified';
  if (d.unreadable) return 'unreadable';
  if (d.review) return 'review';
  if (d.text && String(d.text).trim() !== '') return 'saved';
  return 'pending';
}

function listForFilter() {
  var l = [];
  for (var i = 0; i < TOTAL; i++) {
    if (filter === 'all' || computeStatus(i) === filter) l.push(i);
  }
  return l;
}

function isLocked(i) {
  return ENTRIES[i].verified && !unlocked[i];
}

function move(n) {
  var list = listForFilter();
  if (list.length === 0) { currentIdx = 0; }
  else {
    var pos = list.indexOf(currentIdx);
    if (pos < 0) pos = 0;
    pos = (pos + n + list.length) % list.length;
    currentIdx = list[pos];
  }
  renderEntry();
  renderList();
  updateCounts();
  persist();
}

function saveCurrent() {
  var i = currentIdx;
  if (i < 0 || i >= TOTAL) return;
  if (isLocked(i)) return;                       // never overwrite a locked verified entry
  var d = draftOf(i);
  d.text = gt.value;
  if (String(d.text).trim() !== '') d.review = false;  // a real reading cancels review-needed
  persist();
  updateEntryStatus(i);
  updateCounts();
}

function markUnreadable(i) {
  if (isLocked(i)) { showMsg('Verified entries are locked. Click Modify to change.', 'warn'); return; }
  var d = draftOf(i);
  d.unreadable = !d.unreadable;
  if (d.unreadable) { d.text = ''; d.review = false; showMsg('Marked unreadable (text cleared).', 'bad'); }
  else showMsg('Unreadable cleared.', 'mut');
  persist();
  renderEntry();
  renderList();
  updateCounts();
}

function markReview(i) {
  if (isLocked(i)) { showMsg('Verified entries are locked. Click Modify to change.', 'warn'); return; }
  var d = draftOf(i);
  d.review = !d.review;
  showMsg(d.review ? 'Marked for review.' : 'Review flag cleared.', d.review ? 'warn' : 'mut');
  persist();
  updateEntryStatus(i);
  renderList();
  updateCounts();
}

function verify(i) {
  if (isLocked(i)) { showMsg('This entry is already verified.', 'ok'); return; }
  var d = draftOf(i);
  if (String(d.text || '').trim() === '' && !d.unreadable) {
    showMsg('Type the plate text (or mark unreadable) before verifying.', 'warn');
    return;
  }
  d.verified = true;
  d.review = false;
  showMsg('Marked verified (explicit human action).', 'ok');
  persist();
  updateEntryStatus(i);
  renderList();
  updateCounts();
}

function clearEntry(i) {
  if (isLocked(i)) { showMsg('Verified entries are locked. Click Modify to change.', 'warn'); return; }
  DRAFT[i] = blank(i);
  persist();
  renderEntry();
  renderList();
  updateCounts();
  showMsg('Entry cleared.', 'mut');
}

function updateCounts() {
  var counts = { pending: 0, saved: 0, unreadable: 0, review: 0, verified: 0 };
  for (var i = 0; i < TOTAL; i++) counts[computeStatus(i)]++;
  $('cSaved').textContent = counts.saved;
  $('cUnread').textContent = counts.unreadable;
  $('cReview').textContent = counts.review;
  $('cPending').textContent = counts.pending;
  $('cTot').textContent = TOTAL;
}

function renderList() {
  var listEl = $('list');
  var html = '';
  for (var i = 0; i < TOTAL; i++) {
    var cat = computeStatus(i);
    var d = DRAFT[i] || blank(i);
    var shown = d.text ? String(d.text)
              : (ENTRIES[i].verified ? String(ENTRIES[i].gt_text || '') : '');
    var shortTxt = shown ? shown.slice(0, 40) : '';
    var lk = ENTRIES[i].verified ? ' <span class="lk">\uD83D\uDD12</span>' : '';
    html += '<div class="li ' + cat + '" data-idx="' + i + '">' +
      '<span class="dot"></span>' +
      '<span class="idx">' + (i + 1) + '</span>' +
      '<span class="id">' + IDS[i] + '</span>' +
      '<span class="txt">' + (shortTxt || '\u2014') + '</span>' + lk +
      '</div>';
  }
  listEl.innerHTML = html;
  var items = listEl.querySelectorAll('.li');
  for (var k = 0; k < items.length; k++) {
    if (parseInt(items[k].dataset.idx, 10) === currentIdx) items[k].classList.add('active');
    else items[k].classList.remove('active');
  }
}


function updateEntryStatus(i) {
  if (!mStatus) return;
  var e = ENTRIES[i];
  var d = DRAFT[i] || blank(i);
  var label, cls;
  if (e.verified && !unlocked[i]) { label = 'Verified (locked)'; cls = 'verified'; }
  else if (e.verified && unlocked[i]) { label = 'Verified \u00b7 editing'; cls = 'warn'; }
  else if (d.verified) { label = 'Verified'; cls = 'verified'; }
  else if (d.unreadable) { label = 'Unreadable'; cls = 'bad'; }
  else if (d.review) { label = 'Review needed'; cls = 'rev'; }
  else if (d.text && String(d.text).trim() !== '') { label = 'Saved (unverified)'; cls = 'saved'; }
  else { label = 'Pending'; cls = 'pending'; }
  mStatus.textContent = label;
  mStatus.className = 'tag ' + cls;
}

function renderEntry() {
  var i = currentIdx;
  if (i < 0 || i >= TOTAL) i = currentIdx = 0;
  var e = ENTRIES[i];
  var d = DRAFT[i] || blank(i);
  $('mIdx').textContent = (i + 1) + ' / ' + TOTAL;
  $('mId').textContent = e.id;
  $('mW').textContent = e.crop_w;
  $('mH').textContent = e.crop_h;
  $('mAr').textContent = e.crop_ar ? e.crop_ar.toFixed(2) : '\u2014';
  $('mYolo').textContent = e.yolo_conf ? e.yolo_conf.toFixed(2) : '\u2014';
  $('mSrc').textContent = e.source_image || '\u2014';
  $('mHint').textContent = e.hint || '\u2014';

  // Load the real crop; show a clear error overlay if it is missing.
  img.classList.remove('hidden');
  missing.classList.add('hidden');
  img.onerror = function () {
    img.classList.add('hidden');
    missing.classList.remove('hidden');
    missing.innerHTML = '\u26a0 Could not load crop for <b>' + e.id + '</b>' +
      '<code>' + (e.crop_path || '(no path)') + '</code>';
  };
  img.onload = function () { img.classList.remove('hidden'); missing.classList.add('hidden'); };
  img.src = e.crop_path;
  img.alt = e.id;

  // Ground-truth field: verified entries show their real text, locked by default.
  gt.value = (e.verified && !unlocked[i]) ? String(e.gt_text || '') : String(d.text || '');
  var locked = e.verified && !unlocked[i];
  gt.disabled = locked;
  gt.readOnly = locked;

  var protect = $('protectBox');
  var btnModify = $('btnModify');
  if (e.verified) {
    protect.classList.add('show');
    $('btnSave').disabled = locked;
    $('btnVerify').disabled = true;
    $('btnUnread').disabled = true;
    $('btnReview').disabled = true;
    $('btnClear').disabled = true;
    btnModify.style.display = locked ? '' : 'none';
    if (locked) {
      btnModify.textContent = 'Modify verified';
    } else {
      btnModify.textContent = 'Re-lock';
      protect.classList.remove('show');
      $('btnSave').disabled = false;
    }
  } else {
    protect.classList.remove('show');
    btnModify.style.display = 'none';
    $('btnSave').disabled = false;
    $('btnVerify').disabled = false;
    $('btnUnread').disabled = false;
    $('btnReview').disabled = false;
    $('btnClear').disabled = false;
  }
  updateEntryStatus(i);
  updateCounts();
  renderList();
  renderImage();
}

function setFilter(cat) {
  filter = cat;
  var list = listForFilter();
  if (list.indexOf(currentIdx) < 0) currentIdx = list.length ? list[0] : 0;
  // reflect active filter button
  var btns = document.querySelectorAll('.filters [data-filter]');
  for (var b = 0; b < btns.length; b++) {
    if (btns[b].dataset.filter === cat) btns[b].style.outline = '2px solid var(--acc)';
    else btns[b].style.outline = '';
  }
  renderEntry();
  persist();
}

function jumpTo() {
  var raw = String($('jumpInput').value || '').trim();
  if (!raw) return;
  var target = -1;
  if (/^\d+$/.test(raw)) {
    var n = parseInt(raw, 10);
    if (n >= 1 && n <= TOTAL) target = n - 1;         // 1-based row number
  }
  if (target < 0) {                                    // else treat as an ID
    for (var i = 0; i < TOTAL; i++) {
      if (String(IDS[i]).toUpperCase() === raw.toUpperCase()) { target = i; break; }
    }
  }
  if (target < 0) { showMsg('No entry matches "' + raw + '".', 'warn'); return; }
  currentIdx = target;
  renderEntry();
  persist();
  showMsg('Jumped to ' + IDS[target] + '.', 'mut');
}

function toggleModifyVerified() {
  var i = currentIdx;
  if (!ENTRIES[i].verified) return;
  if (!unlocked[i]) {
    var msg = 'Unlock VERIFIED entry ' + ENTRIES[i].id +
      ' for editing?\n\nCurrent verified text: ' + (ENTRIES[i].gt_text || '(none)') +
      '\n\nIf you Save a different value, the verified ground-truth record will be OVERWRITTEN.';
    if (!window.confirm(msg)) return;
    unlocked[i] = true;
    showMsg('Unlocked for editing. Type the exact text, then Save (with confirmation).', 'warn');
  } else {
    unlocked[i] = false;
    showMsg('Re-locked. Verified value restored in the field.', 'ok');
  }
  renderEntry();
  persist();
}


function zoomBy(f) {
  view.zoom = Math.max(0.12, Math.min(6, view.zoom * f));
  renderImage();
}

function renderImage() {
  var e = ENTRIES[currentIdx];
  if (!e || !e.crop_w) return;
  var availW = stage.clientWidth - 24;
  var availH = stage.clientHeight - 24;
  var cssW = e.crop_w * view.zoom;
  var cssH = cssW * (e.crop_h / e.crop_w);
  if (cssW > availW) { cssW = availW; cssH = cssW * (e.crop_h / e.crop_w); }
  if (cssH > availH) { cssH = availH; cssW = cssH * (e.crop_w / e.crop_h); }
  if (cssW < 8) cssW = 8;
  img.style.width = Math.round(cssW) + 'px';
  img.style.height = 'auto';
  img.style.filter = 'brightness(' + (view.bri / 100) + ') contrast(' + (view.con / 100) +
    ') grayscale(' + (view.gray ? '1' : '0') + ')';
}

function fitImage() {
  var e = ENTRIES[currentIdx];
  if (!e || !e.crop_w) return;
  var availW = stage.clientWidth - 24;
  var availH = stage.clientHeight - 24;
  var scale = Math.min(availW / e.crop_w, availH / e.crop_h, 4);
  if (!isFinite(scale) || scale <= 0) scale = 1;
  view.zoom = scale;
  renderImage();
}

function resetImage() {
  view.zoom = 1; view.bri = 100; view.con = 100; view.gray = false;
  $('bri').value = 100; $('con').value = 100; $('gray').checked = false;
  renderImage();
}

function saveText() {
  var i = currentIdx;
  if (isLocked(i)) { showMsg('Verified entry is locked. Click Modify first.', 'warn'); return; }
  var e = ENTRIES[i];
  var typed = gt.value;
  // If editing a previously verified entry and the value changed, require confirmation.
  if (e.verified && unlocked[i]) {
    var prev = String(e.gt_text || '');
    if (typed !== prev) {
      var msg = 'OVERWRITE verified ground truth for ' + e.id + '?\n\n' +
        'Old (verified): ' + (prev || '(none)') + '\nNew (typed):    ' + (typed || '(empty)');
      if (!window.confirm(msg)) { showMsg('Overwrite cancelled. Verified value kept.', 'warn'); return; }
    }
  }
  saveCurrent();
  showMsg('Saved to browser storage. Use Export to write the canonical answers file.', 'ok');
}

function buildAnswersObject() {
  var answers = {};
  for (var i = 0; i < TOTAL; i++) {
    var e = ENTRIES[i];
    var d = DRAFT[i] || blank(i);
    // Verified entries that were NOT edited export their existing verified text
    // so apply_transcript.py keeps them; edited ones export the new typed value.
    var text = (e.verified && !unlocked[i]) ? String(e.gt_text || '') : String(d.text || '');
    answers[String(i)] = {
      'id': IDS[i],
      'crop_path': e.crop_path,
      'hint': e.hint || '',
      'text': text,
      'unreadable': !!(e.verified && !unlocked[i] ? false : d.unreadable)
    };
  }
  return {
    instructions: 'Manual ground-truth transcriptions for the OCR benchmark. ' +
      'Save this file as review_sheets/transcript_answers.json, then run: ' +
      'python diagnostics/ocr_benchmark/apply_transcript.py',
    answers: answers
  };
}

function exportJSON() {
  var obj = buildAnswersObject();
  try {
    var blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'transcript_answers.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showMsg('Exported transcript_answers.json to Downloads. Move it to ' +
      'diagnostics/ocr_benchmark/review_sheets/ then run apply_transcript.py.', 'ok');
  } catch (err) {
    showMsg('Export failed: ' + err.message, 'bad');
  }
}

function importJSON(file) {
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    try {
      var obj = JSON.parse(reader.result);
      var ans = obj && obj.answers;
      if (!ans || typeof ans !== 'object') throw new Error('no "answers" object');
      var imported = 0, skippedVerified = 0;
      for (var i = 0; i < TOTAL; i++) {
        var a = ans[String(i)];
        if (!a) continue;
        if (ENTRIES[i].verified && !unlocked[i]) {
          // Do not silently overwrite verified records on import.
          if (String(a.text || '') !== String(ENTRIES[i].gt_text || '')) skippedVerified++;
          continue;
        }
        var d = draftOf(i);
        d.text = (typeof a.text === 'string') ? a.text : '';
        d.unreadable = !!a.unreadable;
        imported++;
      }
      persist();
      renderEntry();
      renderList();
      updateCounts();
      var m = 'Imported ' + imported + ' entr' + (imported === 1 ? 'y' : 'ies') + '.';
      if (skippedVerified) m += ' Skipped ' + skippedVerified + ' verified change(s) (locked).';
      showMsg(m, 'ok');
    } catch (err) {
      showMsg('Import failed: ' + err.message, 'bad');
    }
  };
  reader.readAsText(file);
}


function init() {
  restore();
  // Reflect server/file mode tag if present.
  var srv = $('srvtag');
  if (srv) {
    var isFile = (location.protocol === 'file:');
    srv.textContent = isFile ? 'file mode \u00b7 saves to browser + Export' : 'served';
    srv.classList.add(isFile ? 'file' : 'srv');
  }
  // Clamp currentIdx and honour the restored/default filter.
  if (currentIdx < 0 || currentIdx >= TOTAL) currentIdx = 0;
  var list = listForFilter();
  if (list.indexOf(currentIdx) < 0) currentIdx = list.length ? list[0] : 0;

  // Keep the on-screen sliders in sync with restored view state.
  $('bri').value = view.bri;
  $('con').value = view.con;
  $('gray').checked = view.gray;

  renderEntry();
  renderList();
  updateCounts();
  fitImage();
  persist();  // save initial position/filter/view so "where I left off" is stable

  // Warn before leaving with unsaved typed (not yet Saved) text.
  window.addEventListener('beforeunload', function (ev) {
    var i = currentIdx;
    if (i < 0 || i >= TOTAL || isLocked(i)) return;
    var d = DRAFT[i] || blank(i);
    if (gt.value !== String(d.text || '')) {
      ev.preventDefault();
      ev.returnValue = '';
      return '';
    }
  });
}

document.addEventListener('keydown', function (e) {
  var t = e.target;
  var typing = (t && (t.id === 'gt' || t.id === 'jumpInput'));
  if (typing) {
    if (e.ctrlKey || e.metaKey) {
      if (e.key === 's' || e.key === 'S') { e.preventDefault(); saveText(); }
      else if (e.key === 'Enter') { e.preventDefault(); saveText(); move(1); }
    }
    return; // plain keys stay in the field
  }
  if (e.ctrlKey || e.metaKey) {
    if (e.key === 's' || e.key === 'S') { e.preventDefault(); saveText(); }
    return;
  }
  switch (e.key) {
    case 'ArrowLeft':  e.preventDefault(); move(-1); break;
    case 'ArrowRight': e.preventDefault(); move(1); break;
    case 's': case 'S': e.preventDefault(); saveText(); break;
    case 'u': case 'U': e.preventDefault(); markUnreadable(currentIdx); break;
    case 'r': case 'R': e.preventDefault(); markReview(currentIdx); break;
    case 'v': case 'V': e.preventDefault(); verify(currentIdx); break;
    case 'Enter': e.preventDefault(); saveText(); move(1); break;
    case '+': case '=': e.preventDefault(); zoomBy(1.25); break;
    case '-': case '_': e.preventDefault(); zoomBy(0.8); break;
    case '0': e.preventDefault(); fitImage(); break;
  }
});

$('btnSave').addEventListener('click', saveText);
$('btnPrev').addEventListener('click', function () { move(-1); });
$('btnNext').addEventListener('click', function () { move(1); });
$('btnUnread').addEventListener('click', function () { markUnreadable(currentIdx); });
$('btnReview').addEventListener('click', function () { markReview(currentIdx); });
$('btnClear').addEventListener('click', function () { clearEntry(currentIdx); });
$('btnVerify').addEventListener('click', function () { verify(currentIdx); });
$('btnModify').addEventListener('click', toggleModifyVerified);
$('btnExport').addEventListener('click', exportJSON);
$('btnImport').addEventListener('click', function () { $('fileInput').click(); });
$('fileInput').addEventListener('change', function () { importJSON(this.files && this.files[0]); this.value = ''; });
$('btnZoomIn').addEventListener('click', function () { zoomBy(1.25); });
$('btnZoomOut').addEventListener('click', function () { zoomBy(0.8); });
$('btnFit').addEventListener('click', fitImage);
$('btnReset').addEventListener('click', resetImage);
$('btnJump').addEventListener('click', jumpTo);
$('jumpInput').addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); jumpTo(); } });
$('bri').addEventListener('input', function () { view.bri = parseInt(this.value, 10); renderImage(); });
$('con').addEventListener('input', function () { view.con = parseInt(this.value, 10); renderImage(); });
$('gray').addEventListener('change', function () { view.gray = this.checked; renderImage(); });

// Sidebar list: click a row to jump.
$('list').addEventListener('click', function (e) {
  var row = e.target.closest ? e.target.closest('.li') : null;
  if (!row) return;
  currentIdx = parseInt(row.dataset.idx, 10);
  renderEntry();
  persist();
});

// Filter buttons.
var filterBar = document.querySelector('.filters');
if (filterBar) {
  filterBar.addEventListener('click', function (e) {
    var b = e.target.closest ? e.target.closest('[data-filter]') : null;
    if (b) setFilter(b.dataset.filter);
  });
}

window.addEventListener('resize', function () { renderImage(); });

init();

})();

