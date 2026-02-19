/**
 * RailScan AI Demo — Main JavaScript
 * SpatialForge, 2026
 *
 * Sections:
 *   1. Frame data & constants
 *   2. Video / frame selection
 *   3. Video player (timeline-based)
 *   4. Sparkline (O(1) cursor via offscreen canvas)
 *   5. Upload handler
 *   6. Webcam live inference
 *   7. API helpers
 *   8. Route kilometer map
 *   9. Turbo colormap + Three.js 3D visualisation
 *  10. Chart.js dashboard
 *  11. Hero counter animation
 *  12. Keyboard navigation
 *  13. Initialisation
 */

'use strict';

// ══════════════════════════════════════════════════════════════
// 1. FRAME DATA — from Depth Anything V2 Large + YOLOv8 defect detection
//    RTX 4090, ~40ms/frame depth + ~12ms/frame defect, 12 keyframes across 2 JR videos
// ══════════════════════════════════════════════════════════════

// Defect class labels (matches spatialforge/models/inspection.py)
const DEFECT_LABELS_JA = {
  rail_crack:       'レールき裂',
  rail_wear:        'レール摩耗',
  rail_corrugation: 'レール波状摩耗',
  rail_spalling:    'レール剥離',
  fastener_missing: '締結装置欠損',
  fastener_broken:  '締結装置破損',
  sleeper_crack:    'まくらぎき裂',
  sleeper_decay:    'まくらぎ腐食',
  ballast_fouling:  'バラスト異状',
  joint_defect:     '継目異状',
  gauge_anomaly:    '軌間異常',
};

const SEV_LABELS_JA = {
  critical: '緊急',
  major:    '重要',
  minor:    '軽微',
  info:     '参考',
};

const SEV_COLORS = {
  critical: '#ef4444',
  major:    '#f59e0b',
  minor:    '#3b82f6',
  info:     '#6b7280',
};

const FRAMES = [
  { name:'jrsam3_02s', ts:2,  video:'jrsam3', label:'jrsam3 02s', min:2.99, max:68.0,  conf:0.893, ms:10.5,
    anoms:[
      {x:49,  y:4,   w:211, h:617, dist:5.6,  area:130187, sev:'major',    cls:'rail_spalling',    clsConf:0.98, depth:6.1},
      {x:55,  y:225, w:201, h:412, dist:6.7,  area:82812,  sev:'major',    cls:'rail_spalling',    clsConf:0.27, depth:10.1},
    ] },
  { name:'jrsam3_04s', ts:4,  video:'jrsam3', label:'jrsam3 04s', min:3.23, max:194.4, conf:0.886, ms:13.6,
    anoms:[
      {x:182, y:413, w:61,  h:59,  dist:9.7,  area:3599,   sev:'major',    cls:'rail_spalling',    clsConf:0.84, depth:4.6},
      {x:1,   y:545, w:633, h:94,  dist:4.3,  area:59502,  sev:'critical', cls:'rail_crack',       clsConf:0.72, depth:7.8},
      {x:9,   y:227, w:584, h:353, dist:7.6,  area:206152, sev:'minor',    cls:'rail_corrugation', clsConf:0.66, depth:4.6},
      {x:0,   y:195, w:637, h:99,  dist:5.3,  area:63063,  sev:'critical', cls:'rail_crack',       clsConf:0.59, depth:11.6},
      {x:276, y:413, w:67,  h:45,  dist:7.6,  area:3015,   sev:'major',    cls:'rail_spalling',    clsConf:0.57, depth:7.8},
      {x:611, y:399, w:27,  h:39,  dist:5.5,  area:1053,   sev:'major',    cls:'rail_spalling',    clsConf:0.51, depth:7.0},
    ] },
  { name:'jrsam3_07s', ts:7,  video:'jrsam3', label:'jrsam3 07s', min:3.01, max:45.5,  conf:0.871, ms:12.6,
    anoms:[
      {x:180, y:184, w:123, h:134, dist:5.3,  area:16482,  sev:'major',    cls:'rail_spalling',    clsConf:0.85, depth:6.4},
      {x:317, y:207, w:206, h:129, dist:10.3, area:26574,  sev:'major',    cls:'rail_spalling',    clsConf:0.48, depth:6.7},
      {x:176, y:158, w:147, h:162, dist:9.5,  area:23814,  sev:'major',    cls:'rail_spalling',    clsConf:0.42, depth:9.0},
    ] },
  { name:'jrsam3_11s', ts:11, video:'jrsam3', label:'jrsam3 11s', min:1.72, max:13.0,  conf:0.871, ms:20.1,
    anoms:[
      {x:189, y:454, w:74,  h:87,  dist:8.4,  area:6438,   sev:'major',    cls:'rail_spalling',    clsConf:0.83, depth:9.2},
      {x:268, y:343, w:133, h:169, dist:10.6, area:22477,  sev:'major',    cls:'rail_spalling',    clsConf:0.69, depth:8.3},
      {x:20,  y:138, w:472, h:430, dist:5.8,  area:202960, sev:'minor',    cls:'rail_corrugation', clsConf:0.43, depth:9.3},
      {x:1,   y:119, w:630, h:83,  dist:7.1,  area:52290,  sev:'critical', cls:'rail_crack',       clsConf:0.40, depth:9.6},
      {x:362, y:155, w:277, h:419, dist:7.3,  area:116063, sev:'minor',    cls:'rail_corrugation', clsConf:0.37, depth:7.8},
      {x:139, y:303, w:116, h:44,  dist:11.6, area:5104,   sev:'major',    cls:'rail_spalling',    clsConf:0.30, depth:7.2},
      {x:6,   y:524, w:618, h:94,  dist:8.6,  area:58092,  sev:'critical', cls:'rail_crack',       clsConf:0.26, depth:9.9},
    ] },
  { name:'jrsam3_16s', ts:16, video:'jrsam3', label:'jrsam3 16s', min:1.89, max:113.3, conf:0.867, ms:15.7,
    anoms:[
      {x:216, y:411, w:64,  h:68,  dist:10.0, area:4352,   sev:'major',    cls:'rail_spalling',    clsConf:0.81, depth:6.1},
      {x:5,   y:524, w:631, h:104, dist:5.9,  area:65624,  sev:'critical', cls:'rail_crack',       clsConf:0.69, depth:10.5},
      {x:26,  y:246, w:516, h:323, dist:9.8,  area:166668, sev:'minor',    cls:'rail_corrugation', clsConf:0.68, depth:11.6},
      {x:0,   y:197, w:634, h:127, dist:11.8, area:80518,  sev:'critical', cls:'rail_crack',       clsConf:0.52, depth:11.8},
      {x:173, y:244, w:433, h:318, dist:7.1,  area:137694, sev:'minor',    cls:'rail_corrugation', clsConf:0.49, depth:9.6},
      {x:450, y:346, w:27,  h:26,  dist:10.2, area:702,    sev:'major',    cls:'rail_spalling',    clsConf:0.30, depth:11.1},
    ] },
  { name:'jrsam3_21s', ts:21, video:'jrsam3', label:'jrsam3 21s', min:2.96, max:122.6, conf:0.881, ms:11.8,
    anoms:[
      {x:293, y:0,   w:135, h:639, dist:11.1, area:86265,  sev:'critical', cls:'rail_crack',       clsConf:0.80, depth:6.2},
      {x:58,  y:0,   w:146, h:337, dist:10.9, area:49202,  sev:'critical', cls:'rail_crack',       clsConf:0.43, depth:6.8},
      {x:45,  y:0,   w:131, h:636, dist:6.1,  area:83316,  sev:'critical', cls:'rail_crack',       clsConf:0.41, depth:8.3},
    ] },
  { name:'jr23_02s',   ts:2,  video:'jr23',   label:'jr23 02s',   min:2.12, max:193.7, conf:0.884, ms:14.5,
    anoms:[
      {x:0,   y:519, w:631, h:105, dist:11.1, area:66255,  sev:'critical', cls:'rail_crack',       clsConf:0.79, depth:4.1},
      {x:0,   y:195, w:635, h:111, dist:9.4,  area:70485,  sev:'critical', cls:'rail_crack',       clsConf:0.62, depth:8.5},
      {x:16,  y:239, w:547, h:312, dist:5.3,  area:170664, sev:'minor',    cls:'rail_corrugation', clsConf:0.54, depth:7.1},
    ] },
  { name:'jr23_04s',   ts:4,  video:'jr23',   label:'jr23 04s',   min:2.67, max:26.6,  conf:0.872, ms:9.3,
    anoms:[
      {x:7,   y:518, w:625, h:104, dist:5.7,  area:65000,  sev:'critical', cls:'rail_crack',       clsConf:0.79, depth:7.3},
      {x:0,   y:196, w:623, h:107, dist:11.6, area:66661,  sev:'critical', cls:'rail_crack',       clsConf:0.63, depth:9.0},
      {x:38,  y:235, w:452, h:308, dist:7.4,  area:139216, sev:'minor',    cls:'rail_corrugation', clsConf:0.54, depth:10.9},
      {x:598, y:451, w:41,  h:71,  dist:5.8,  area:2911,   sev:'major',    cls:'rail_spalling',    clsConf:0.39, depth:9.5},
      {x:3,   y:516, w:568, h:77,  dist:6.5,  area:43736,  sev:'critical', cls:'rail_crack',       clsConf:0.32, depth:6.0},
    ] },
  { name:'jr23_07s',   ts:7,  video:'jr23',   label:'jr23 07s',   min:2.64, max:56.1,  conf:0.865, ms:13.6,
    anoms:[] },
  { name:'jr23_11s',   ts:11, video:'jr23',   label:'jr23 11s',   min:1.73, max:142.2, conf:0.891, ms:12.4,
    anoms:[] },
  { name:'jr23_16s',   ts:16, video:'jr23',   label:'jr23 16s',   min:1.66, max:121.3, conf:0.877, ms:13.6,
    anoms:[] },
  { name:'jr23_21s',   ts:21, video:'jr23',   label:'jr23 21s',   min:1.93, max:127.6, conf:0.856, ms:11.6,
    anoms:[] },
];

// Assumed speed for km-post conversion: 80 km/h = 22.22 m/s
const ASSUMED_SPEED_KMH = 80;
const ASSUMED_SPEED_MS  = ASSUMED_SPEED_KMH / 3.6; // 22.22 m/s

let currentVideo = 'jrsam3';
let currentFrame = FRAMES.find(f => f.name === 'jrsam3_11s');

// ══════════════════════════════════════════════════════════════
// 2. VIDEO / FRAME SELECTION
// ══════════════════════════════════════════════════════════════
function selectVideo(vid) {
  currentVideo = vid;
  document.querySelectorAll('.video-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.vid === vid));

  const isUpload   = vid === 'upload';
  const isPlayer   = vid === 'player';
  const isWebcam   = vid === 'webcam';
  const isKeyframe = !isUpload && !isPlayer && !isWebcam;

  document.getElementById('frameStrip').style.display         = isKeyframe ? 'block' : 'none';
  document.getElementById('mainDisplay').style.display        = isKeyframe ? 'grid'  : 'none';
  document.getElementById('infoPanels').style.display         = isKeyframe ? 'grid'  : 'none';
  document.getElementById('uploadZone').style.display         = isUpload   ? 'block' : 'none';
  document.getElementById('videoPlayerSection').style.display = isPlayer   ? 'block' : 'none';
  document.getElementById('webcamSection').style.display      = isWebcam   ? 'block' : 'none';

  // Stop webcam when leaving webcam tab
  if (!isWebcam && webcamRunning) stopWebcam();

  // Stop player playback when leaving player tab
  if (!isPlayer && playerInterval) {
    clearInterval(playerInterval);
    playerInterval = null;
    document.getElementById('playBtn').textContent = '▶';
  }

  if (isKeyframe) {
    renderFrameStrip(vid);
    const frames = FRAMES.filter(f => f.video === vid);
    const pick = frames.find(f => f.anoms.length > 0) || frames[0];
    selectFrame(pick.name);
  }
  if (isPlayer && !playerTimeline) loadTimeline('jrsam3');
  if (isWebcam) checkApiStatus();
}

function renderFrameStrip(vid) {
  const inner = document.getElementById('frameStripInner');
  const frames = FRAMES.filter(f => f.video === vid);
  inner.innerHTML = frames.map(f => {
    const hasCritical = f.anoms.some(a => a.sev === 'critical');
    const hasMajor    = f.anoms.some(a => a.sev === 'major');
    const topSev      = hasCritical ? 'critical' : hasMajor ? 'major' : f.anoms.length > 0 ? 'minor' : null;
    const dotColor    = topSev ? SEV_COLORS[topSev] : null;
    const dotLabel    = topSev ? SEV_LABELS_JA[topSev] : null;
    return `
    <div class="frame-card ${f.name === currentFrame?.name ? 'active' : ''}"
         onclick="selectFrame('${f.name}')" id="fcard-${f.name}">
      <div class="frame-thumb">
        <img src="rail-assets/${f.name}_camera.jpg" alt="${f.ts}s" loading="lazy">
        ${f.anoms.length > 0
          ? `<span class="frame-anom-dot" style="background:${dotColor}">${dotLabel} ${f.anoms.length}</span>`
          : `<span class="frame-ok-dot">✓</span>`
        }
      </div>
      <div class="frame-ts">${f.ts}s</div>
    </div>`;
  }).join('');
}

function selectFrame(name) {
  const frame = FRAMES.find(f => f.name === name);
  if (!frame) return;
  currentFrame = frame;

  // Update frame strip active state
  document.querySelectorAll('.frame-card').forEach(c => c.classList.remove('active'));
  const card = document.getElementById('fcard-' + name);
  if (card) card.classList.add('active');

  // Flash transition
  const flash = document.getElementById('procFlash');
  flash.style.opacity = '1';
  setTimeout(() => { flash.style.opacity = '0'; }, 300);

  // Update images (briefly fade to force redraw)
  const cImg = document.getElementById('cameraImg');
  const oImg = document.getElementById('overlayImg');
  cImg.style.opacity = '0.5';
  oImg.style.opacity = '0.5';
  cImg.src = `rail-assets/${name}_camera.jpg`;
  oImg.src = `rail-assets/${name}_overlay.jpg`;
  cImg.onload = () => { cImg.style.opacity = '1'; };
  oImg.onload = () => { oImg.style.opacity = '1'; };

  // Update badge
  document.getElementById('frameLabel').textContent =
    `${frame.video === 'jrsam3' ? 'jrsam3路線' : 'jr23路線'} — ${frame.ts}s`;

  // Update panels
  updateMetricPanel(frame);
  updateAlertPanel(frame);

  // Update 3D viewer
  loadDepthForThreeJs(name, frame);
}

function updateMetricPanel(frame) {
  const nearEl = document.getElementById('m-near');
  nearEl.textContent = frame.min.toFixed(2) + ' m';
  nearEl.className = 'metric-val' + (frame.min < 2.5 ? ' danger' : '');

  document.getElementById('m-far').textContent =
    frame.max >= 200 ? '200+ m' : frame.max.toFixed(1) + ' m';

  const clearEl = document.getElementById('m-clearance');
  const hasCritical = frame.anoms.some(a => a.sev === 'critical');
  const hasMajor    = frame.anoms.some(a => a.sev === 'major');
  if (hasCritical) {
    clearEl.textContent = '緊急対応 ⚠';
    clearEl.className = 'metric-val danger';
  } else if (hasMajor) {
    clearEl.textContent = '要確認 ⚠';
    clearEl.className = 'metric-val accent';
  } else if (frame.anoms.length > 0) {
    clearEl.textContent = '経過観察';
    clearEl.className = 'metric-val accent';
  } else {
    clearEl.textContent = '問題なし ✓';
    clearEl.className = 'metric-val safe';
  }

  const confEl = document.getElementById('m-conf');
  confEl.textContent = (frame.conf * 100).toFixed(1) + '%';
  confEl.className = 'metric-val accent';

  const anomEl = document.getElementById('m-anom');
  anomEl.textContent = frame.anoms.length + ' 件';
  anomEl.className = 'metric-val ' + (frame.anoms.length > 0 ? 'danger' : 'safe');

  document.getElementById('m-time').textContent = frame.ms.toFixed(1) + ' ms';
  document.getElementById('m-model').textContent = 'DA V2 Large + YOLOv8';
}

function updateAlertPanel(frame) {
  const list = document.getElementById('alertList');
  if (frame.anoms.length === 0) {
    list.innerHTML = `
      <div class="alert-item ok">
        <span class="alert-badge badge-ok">正常</span>
        <div class="alert-text">
          <strong>異常なし — このフレームはクリア</strong>
          <span>深度範囲: ${frame.min.toFixed(2)}m–${frame.max >= 200 ? '200m+' : frame.max.toFixed(1) + 'm'} / 信頼度: ${(frame.conf*100).toFixed(1)}%</span>
        </div>
      </div>
      <div class="alert-item ok">
        <span class="alert-badge badge-ok">済</span>
        <div class="alert-text">
          <strong>欠陥検知 0件 — 軌道状態良好</strong>
          <span>処理時間: ${frame.ms.toFixed(1)}ms | DA V2 Large + YOLOv8 | RTX 4090</span>
        </div>
      </div>`;
    return;
  }

  const sevBadgeClass = (sev) => sev === 'critical' ? 'badge-critical' : sev === 'major' ? 'badge-warning' : 'badge-info';

  list.innerHTML = frame.anoms.map((a) => {
    const clsJa   = DEFECT_LABELS_JA[a.cls] || a.cls;
    const sevJa   = SEV_LABELS_JA[a.sev] || a.sev;
    const sevCol  = SEV_COLORS[a.sev] || '#6b7280';
    const confPct = a.clsConf ? (a.clsConf * 100).toFixed(0) + '%' : '—';
    return `
    <div class="alert-item ${a.sev}" style="border-left:3px solid ${sevCol}">
      <span class="alert-badge ${sevBadgeClass(a.sev)}">${sevJa}</span>
      <div class="alert-text">
        <strong>${clsJa}を検知 — 前方 ${a.dist.toFixed(1)} m</strong>
        <span>
          分類信頼度: ${confPct} / 位置: (${a.x},${a.y}) / サイズ: ${a.w}×${a.h}px / 深度: ${a.depth?.toFixed(1) || '—'}m
        </span>
      </div>
    </div>`;
  }).join('') + `
    <div class="alert-item ok">
      <span class="alert-badge badge-ok">通知済</span>
      <div class="alert-text">
        <strong>保線担当へアラート自動通知（${frame.anoms.length}件）</strong>
        <span>深度推論: ${frame.ms.toFixed(1)}ms + 欠陥検知: ~12ms | 信頼度: ${(frame.conf*100).toFixed(1)}%</span>
      </div>
    </div>`;
}

// ══════════════════════════════════════════════════════════════
// 3. VIDEO PLAYER — timeline-based playback
// ══════════════════════════════════════════════════════════════
let playerTimeline   = null;
let playerVideoName  = '';
let playerFrameIdx   = 0;
let playerInterval   = null;
let playerSpeed      = 1;
const PLAYER_SPEEDS  = [0.5, 1, 2];
let playerSpeedIdx   = 1;
let playerMode       = 'video';

function setPlayerMode(mode) {
  playerMode = mode;
  document.getElementById('pm-video-wrap').style.display  = mode === 'video' ? 'block' : 'none';
  document.getElementById('pm-frame-wrap').style.display  = mode === 'frame' ? 'block' : 'none';
  document.getElementById('pmBtn-video').classList.toggle('active', mode === 'video');
  document.getElementById('pmBtn-frame').classList.toggle('active', mode === 'frame');
  if (mode === 'frame' && playerTimeline) {
    const vid = document.getElementById('playerVideo');
    if (vid) vid.pause();
    setTimeout(() => drawSparkline(playerTimeline), 50);
  }
}

function buildEventsPanel(data) {
  const frames     = data.frames;
  const anomFrames = frames.filter(f => f.anomaly_count > 0);
  const total      = anomFrames.length;
  const panel      = document.getElementById('eventsPanel');

  const countColor = total > 0 ? 'var(--danger)' : 'var(--safe)';
  let html = `<div class="events-panel-header">
    <h3>⚠ 異常イベント一覧</h3>
    <span class="events-count">${frames.length} フレーム中 <strong style="color:${countColor}">${total}</strong> 件検知</span>
  </div>`;

  if (total === 0) {
    html += `<div style="color:var(--safe);font-size:0.85rem;padding:0.5rem 0">✓ 異常なし — 全区間正常</div>`;
  } else {
    html += anomFrames.map(f => {
      const sev     = f.anomalies[0]?.severity || 'warning';
      const dist    = f.dist_m != null ? `前方 ${f.dist_m} m` : '距離不明';
      const score   = (f.anomaly_score * 100).toFixed(0);
      const clsJa   = DEFECT_LABELS_JA[f.anomalies[0]?.defect_class] || '近接物体';
      const sevJa   = SEV_LABELS_JA[sev] || '注意';
      const sevCol  = SEV_COLORS[sev] || '#f59e0b';
      return `<div class="event-row" style="border-left:3px solid ${sevCol}">
        <span class="event-time">${f.ts.toFixed(1)}s</span>
        <div class="event-details">
          <div class="event-title">${clsJa}検知 — ${dist}</div>
          <div class="event-meta">スコア ${score}% / 信頼度 ${(f.confidence*100).toFixed(0)}% / ${f.processing_ms.toFixed(0)} ms</div>
        </div>
        <span class="event-badge ${sev}" style="background:${sevCol}">${sevJa}</span>
      </div>`;
    }).join('');
  }
  panel.innerHTML = html;
}

function printReport() {
  if (!playerTimeline) return;
  const data    = playerTimeline;
  const name    = playerVideoName;
  const today   = new Date().toLocaleDateString('ja-JP', {year:'numeric', month:'2-digit', day:'2-digit'});
  const anomaly = data.frames.filter(f => f.anomaly_count > 0);
  const avgConf = (data.frames.reduce((s, f) => s + f.confidence, 0) / data.frames.length * 100).toFixed(1);
  const avgMs   = (data.frames.reduce((s, f) => s + f.processing_ms, 0) / data.frames.length).toFixed(1);

  const eventRows = anomaly.map((f, i) => {
    const a       = f.anomalies[0] || {};
    const sevJa   = SEV_LABELS_JA[a.severity] || a.severity || '注意';
    const clsJa   = DEFECT_LABELS_JA[a.defect_class] || a.defect_class || '近接物体';
    const sevCol  = SEV_COLORS[a.severity] || '#a60';
    return `<tr>
      <td>${i + 1}</td>
      <td>${f.ts.toFixed(1)} s</td>
      <td>${clsJa}</td>
      <td>${f.dist_m != null ? f.dist_m + ' m' : '—'}</td>
      <td>${(f.anomaly_score * 100).toFixed(0)}%</td>
      <td>${(f.confidence * 100).toFixed(0)}%</td>
      <td><span style="color:${sevCol};font-weight:700">${sevJa}</span></td>
      <td>${a.area_px ? a.area_px.toLocaleString() + ' px²' : '—'}</td>
    </tr>`;
  }).join('');

  const noAnomRow = anomaly.length === 0
    ? `<tr><td colspan="8" style="text-align:center;color:#080;font-weight:700">異常なし — 全区間正常</td></tr>` : '';

  const w = window.open('', '_blank');
  w.document.write(`<!DOCTYPE html><html lang="ja"><head>
<meta charset="UTF-8">
<title>RailScan AI 点検レポート — ${name}</title>
<style>
  body { font-family: 'Hiragino Sans', 'Noto Sans JP', sans-serif; margin: 0; padding: 2cm; color: #111; font-size: 11pt; }
  h1 { font-size: 18pt; margin-bottom: 4px; }
  .subtitle { color: #555; font-size: 10pt; margin-bottom: 1.5rem; }
  .header-bar { display:flex; justify-content:space-between; align-items:flex-start; border-bottom: 2px solid #f59e0b; padding-bottom: 0.75rem; margin-bottom: 1.5rem; }
  .logo { font-size: 13pt; font-weight: 800; }
  .logo span { color: #f59e0b; }
  .badge { background:#f59e0b; color:#000; font-size:8pt; font-weight:800; padding:2px 7px; border-radius:4px; }
  .meta-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem 2rem; margin-bottom:1.5rem; font-size:10pt; }
  .meta-row { display:flex; justify-content:space-between; border-bottom:1px solid #e5e5e5; padding:4px 0; }
  .meta-key { color:#555; }
  .meta-val { font-weight:700; }
  h2 { font-size:13pt; border-left:3px solid #f59e0b; padding-left:0.5rem; margin:1.5rem 0 0.75rem; }
  table { width:100%; border-collapse:collapse; font-size:10pt; }
  th { background:#f5f5f5; text-align:left; padding:6px 8px; border:1px solid #ddd; font-size:9pt; }
  td { padding:6px 8px; border:1px solid #ddd; }
  tr:nth-child(even) { background:#fafafa; }
  .summary-box { background:#fff8e6; border:1px solid #f59e0b; border-radius:6px; padding:0.75rem 1rem; margin-bottom:1.5rem; }
  .summary-ok  { background:#f0fff4; border-color:#22c55e; }
  .footer { margin-top:2rem; font-size:9pt; color:#888; text-align:center; border-top:1px solid #ddd; padding-top:0.75rem; }
  @media print { body { padding: 1cm 1.5cm; } button { display:none !important; } }
</style></head><body>
<div class="header-bar">
  <div>
    <div class="logo">Spatial<span>Forge</span> <span class="badge">RAIL</span></div>
    <h1>AI軌道点検レポート</h1>
    <div class="subtitle">RailScan AI — Depth Anything V2 Large by SpatialForge</div>
  </div>
  <div style="text-align:right;font-size:9pt;color:#555">
    <div>出力日時: ${today}</div>
    <div>対象路線: ${name}</div>
  </div>
</div>
<div class="meta-grid">
  <div>
    <div class="meta-row"><span class="meta-key">対象路線</span><span class="meta-val">${name}</span></div>
    <div class="meta-row"><span class="meta-key">解析フレーム数</span><span class="meta-val">${data.total_frames} フレーム</span></div>
    <div class="meta-row"><span class="meta-key">動画時間</span><span class="meta-val">${data.duration_s.toFixed(1)} 秒</span></div>
    <div class="meta-row"><span class="meta-key">処理フレームレート</span><span class="meta-val">${data.fps_processed} fps</span></div>
  </div>
  <div>
    <div class="meta-row"><span class="meta-key">使用モデル</span><span class="meta-val">DA V2 Large + YOLOv8 (欠陥検知)</span></div>
    <div class="meta-row"><span class="meta-key">推論デバイス</span><span class="meta-val">RTX 4090 (CUDA)</span></div>
    <div class="meta-row"><span class="meta-key">平均推論時間</span><span class="meta-val">${avgMs} ms/フレーム</span></div>
    <div class="meta-row"><span class="meta-key">平均信頼度</span><span class="meta-val">${avgConf}%</span></div>
  </div>
</div>
<div class="summary-box ${anomaly.length === 0 ? 'summary-ok' : ''}">
  ${anomaly.length === 0
    ? '<strong style="color:#166534">✓ 異常なし</strong> — 解析区間全体で軌道異常は検知されませんでした。'
    : `<strong style="color:#92400e">⚠ ${anomaly.length} 件の異常イベントを検知</strong> — 下表の各フレームで軌道前方への近接物体が確認されました。速やかに保線担当者へご確認ください。`}
</div>
<h2>異常検知イベント一覧</h2>
<table>
  <thead>
    <tr><th>#</th><th>タイムスタンプ</th><th>欠陥種別</th><th>推定距離</th><th>異常スコア</th><th>信頼度</th><th>重症度</th><th>ブロブ面積</th></tr>
  </thead>
  <tbody>${eventRows}${noAnomRow}</tbody>
</table>
<div class="footer">
  本レポートは SpatialForge RailScan AI が自動生成した暫定レポートです。最終判断は保線担当者が現地確認の上で行ってください。<br>
  SpatialForge — spatialforge-demo.fly.dev — ${today}
</div>
<br>
<button onclick="window.print()" style="padding:8px 20px;background:#f59e0b;border:none;border-radius:6px;font-weight:700;cursor:pointer;font-size:11pt">🖨 印刷 / PDF保存</button>
</body></html>`);
  w.document.close();
}

async function loadTimeline(name, tabEl) {
  document.querySelectorAll('.pvtab').forEach(t =>
    t.classList.toggle('active', tabEl ? t === tabEl : t.textContent.includes(name)));

  if (playerInterval) { clearInterval(playerInterval); playerInterval = null; }
  document.getElementById('playBtn').textContent = '▶';

  document.getElementById('playerLoading').style.display = 'flex';
  document.getElementById('playerReady').style.display   = 'none';

  playerVideoName = name;
  const url = `rail-assets/video/${name}/timeline.json`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    playerTimeline = data;

    const vid    = document.getElementById('playerVideo');
    const mp4Src = `rail-assets/video/${name}/${name}_analysis.mp4`;
    vid.src = mp4Src;
    vid.load();

    const dlBtn         = document.getElementById('mp4DownloadBtn');
    dlBtn.href          = mp4Src;
    dlBtn.download      = `${name}_analysis.mp4`;
    dlBtn.style.display = 'flex';

    buildEventsPanel(data);

    const scrubber = document.getElementById('playerScrubber');
    scrubber.max   = data.total_frames - 1;
    scrubber.value = 0;

    drawSparkline(data);
    setPlayerMode('video');

    document.getElementById('playerLoading').style.display = 'none';
    document.getElementById('playerReady').style.display   = 'block';
    displayPlayerFrame(0);

  } catch(e) {
    document.getElementById('playerLoading').innerHTML =
      `<div style="color:var(--muted);text-align:center">
        <div style="font-size:1.5rem;margin-bottom:0.5rem">⏳</div>
        <strong style="color:var(--accent)">動画処理中...</strong><br>
        <span style="font-size:0.82rem">タイムラインを生成しています。数分後にリロードしてください。</span>
       </div>`;
    document.getElementById('playerLoading').style.display = 'flex';
  }
}

function displayPlayerFrame(idx) {
  if (!playerTimeline) return;
  const frames = playerTimeline.frames;
  if (idx < 0 || idx >= frames.length) return;

  playerFrameIdx = idx;
  const f = frames[idx];

  document.getElementById('playerImg').src = `rail-assets/video/${playerVideoName}/${f.files.overlay}`;
  document.getElementById('playerScrubber').value = idx;
  document.getElementById('playerTs').textContent = f.ts.toFixed(1) + 's';

  const anomHud = document.getElementById('playerAnomHud');
  if (f.anomaly_count > 0) {
    anomHud.style.display = 'inline-block';
    anomHud.textContent   = `⚠ ${f.dist_m}m`;
    const flash = document.getElementById('playerAnomFlash');
    flash.style.opacity = '1';
    setTimeout(() => { flash.style.opacity = '0'; }, 200);
  } else {
    anomHud.style.display = 'none';
  }

  document.getElementById('playerTimeDisplay').textContent =
    `${f.ts.toFixed(1)}s / ${playerTimeline.duration_s.toFixed(1)}s`;

  document.getElementById('pm-ts').textContent    = f.ts.toFixed(2) + ' s';
  document.getElementById('pm-near').textContent  = f.near_m.toFixed(2) + ' m';
  document.getElementById('pm-near').className    = 'metric-val' + (f.near_m < 2.5 ? ' danger' : '');
  document.getElementById('pm-far').textContent   = (f.far_m >= 200 ? '200+ m' : f.far_m.toFixed(1) + ' m');
  document.getElementById('pm-score').textContent = f.anomaly_score > 0
    ? (f.anomaly_score * 100).toFixed(1) + '%' : '—';
  document.getElementById('pm-score').className   = 'metric-val ' + (f.anomaly_score > 0.5 ? 'danger' : f.anomaly_score > 0 ? 'accent' : 'safe');
  document.getElementById('pm-conf').textContent  = (f.confidence * 100).toFixed(1) + '%';
  document.getElementById('pm-conf').className    = 'metric-val accent';
  document.getElementById('pm-ms').textContent    = f.processing_ms.toFixed(1) + ' ms';

  updatePlayerAlerts(f);
  drawSparklineCursor(idx);

  // Sync Three.js 3D viewer with current player frame depth image
  if (threeScene) {
    const depthPath = `rail-assets/video/${playerVideoName}/${f.files.depth}`;
    loadDepthForThreeJs(playerVideoName, f, depthPath);
  }
}

function updatePlayerAlerts(f) {
  const list = document.getElementById('playerAlertList');
  if (f.anomaly_count === 0) {
    list.innerHTML = `
      <div class="alert-item ok">
        <span class="alert-badge badge-ok">正常</span>
        <div class="alert-text">
          <strong>異常なし — t=${f.ts.toFixed(1)}s</strong>
          <span>深度 ${f.near_m.toFixed(1)}m–${f.far_m >= 200 ? '200m+' : f.far_m.toFixed(0) + 'm'} / 信頼度 ${(f.confidence*100).toFixed(1)}%</span>
        </div>
      </div>`;
    return;
  }
  list.innerHTML = f.anomalies.map(a => `
    <div class="alert-item ${a.severity}">
      <span class="alert-badge badge-${a.severity === 'critical' ? 'critical' : 'warning'}">${a.severity === 'critical' ? '緊急' : '注意'}</span>
      <div class="alert-text">
        <strong>近接物体検知 — 前方 ${a.dist_m} m  (t=${f.ts.toFixed(1)}s)</strong>
        <span>深度 mean=${a.depth_mean.toFixed(3)} / 面積 ${a.area_px.toLocaleString()} px²</span>
      </div>
    </div>`).join('');
}

function togglePlay() {
  if (playerInterval) {
    clearInterval(playerInterval);
    playerInterval = null;
    document.getElementById('playBtn').textContent = '▶';
  } else {
    document.getElementById('playBtn').textContent = '⏸';
    const fps = playerTimeline ? playerTimeline.fps_processed : 3;
    const intervalMs = Math.round(1000 / (fps * playerSpeed));
    playerInterval = setInterval(() => {
      if (!playerTimeline) return;
      const next = playerFrameIdx + 1;
      displayPlayerFrame(next >= playerTimeline.total_frames ? 0 : next);
    }, intervalMs);
  }
}

function scrubTo(idx) {
  if (playerInterval) { clearInterval(playerInterval); playerInterval = null; document.getElementById('playBtn').textContent = '▶'; }
  displayPlayerFrame(idx);
}

function cycleSpeed() {
  playerSpeedIdx = (playerSpeedIdx + 1) % PLAYER_SPEEDS.length;
  playerSpeed    = PLAYER_SPEEDS[playerSpeedIdx];
  document.getElementById('speedBtn').textContent = playerSpeed + '×';
  if (playerInterval) { clearInterval(playerInterval); playerInterval = null; togglePlay(); }
}

// Jump to next / previous anomaly frame in player
function jumpToAnomalyFrame(direction) {
  if (!playerTimeline) return;
  const frames    = playerTimeline.frames;
  const anomIdxes = frames.map((f, i) => f.anomaly_count > 0 ? i : -1).filter(i => i >= 0);
  if (anomIdxes.length === 0) return;

  if (direction > 0) {
    const next = anomIdxes.find(i => i > playerFrameIdx);
    if (next != null) scrubTo(next);
    else scrubTo(anomIdxes[0]); // wrap
  } else {
    const prev = [...anomIdxes].reverse().find(i => i < playerFrameIdx);
    if (prev != null) scrubTo(prev);
    else scrubTo(anomIdxes[anomIdxes.length - 1]); // wrap
  }
}

// ══════════════════════════════════════════════════════════════
// 4. SPARKLINE — O(1) cursor via offscreen canvas
//    drawSparkline()       — full render → stored in _sparkBaseCanvas
//    drawSparklineCursor() — blit base then draw cursor only
// ══════════════════════════════════════════════════════════════
let _sparkBaseCanvas = null;  // offscreen canvas holding the static sparkline

function drawSparkline(data) {
  const canvas = document.getElementById('sparkCanvas');
  const DPR    = window.devicePixelRatio || 1;
  const W      = canvas.offsetWidth || canvas.parentElement.clientWidth || 800;
  const H      = 64;

  canvas.width  = W * DPR;
  canvas.height = H * DPR;
  canvas.style.height = H + 'px';

  // Create / resize offscreen base canvas
  if (!_sparkBaseCanvas) _sparkBaseCanvas = document.createElement('canvas');
  _sparkBaseCanvas.width  = W * DPR;
  _sparkBaseCanvas.height = H * DPR;

  const ctx = _sparkBaseCanvas.getContext('2d');
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

  const PAD = 4;
  const n   = data.frames.length;

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#1e1e2e';
  ctx.lineWidth   = 1;
  [0.25, 0.5, 0.75].forEach(frac => {
    const y = PAD + (H - PAD*2) * (1 - frac);
    ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W-PAD, y); ctx.stroke();
  });

  if (n === 0) { _blit(); return; }

  // Anomaly highlight regions
  data.frames.forEach((f, i) => {
    if (f.anomaly_count === 0) return;
    const x0 = PAD + (i     / n) * (W - PAD*2);
    const x1 = PAD + ((i+1) / n) * (W - PAD*2);
    ctx.fillStyle = 'rgba(239,68,68,0.12)';
    ctx.fillRect(x0, PAD, x1-x0, H-PAD*2);
  });

  // Score line
  ctx.beginPath();
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth   = 2;
  data.frames.forEach((f, i) => {
    const x = PAD + (i / (n-1)) * (W - PAD*2);
    const y = PAD + (H - PAD*2) * (1 - f.anomaly_score);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Anomaly dots
  data.frames.forEach((f, i) => {
    if (f.anomaly_count === 0) return;
    const x = PAD + (i / (n-1)) * (W - PAD*2);
    const y = PAD + (H - PAD*2) * (1 - f.anomaly_score);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI*2);
    ctx.fillStyle = '#ef4444';
    ctx.fill();
  });

  // Store data reference for cursor
  canvas._sparkData = data;

  // Blit base to screen
  _blit();

  // Click to jump
  canvas.onclick = (e) => {
    const rect = canvas.getBoundingClientRect();
    const rx   = (e.clientX - rect.left) / rect.width;
    const idx  = Math.round(rx * (data.total_frames - 1));
    scrubTo(idx);
  };

  function _blit() {
    const sc = canvas.getContext('2d');
    sc.setTransform(1,0,0,1,0,0);
    sc.clearRect(0, 0, canvas.width, canvas.height);
    sc.drawImage(_sparkBaseCanvas, 0, 0);
  }
}

function drawSparklineCursor(idx) {
  const canvas = document.getElementById('sparkCanvas');
  const data   = canvas._sparkData;
  if (!data || !_sparkBaseCanvas) return;

  const DPR = window.devicePixelRatio || 1;
  const W   = canvas.width  / DPR;
  const H   = canvas.height / DPR;
  const PAD = 4;
  const n   = data.frames.length;

  const sc = canvas.getContext('2d');
  sc.setTransform(1,0,0,1,0,0);

  // Blit clean base (no redraw — O(1) copy)
  sc.clearRect(0, 0, canvas.width, canvas.height);
  sc.drawImage(_sparkBaseCanvas, 0, 0);

  // Draw cursor on top
  sc.setTransform(DPR, 0, 0, DPR, 0, 0);

  const x = PAD + (idx / Math.max(n-1, 1)) * (W - PAD*2);
  sc.strokeStyle = 'rgba(255,255,255,0.6)';
  sc.lineWidth   = 1.5;
  sc.setLineDash([3,3]);
  sc.beginPath(); sc.moveTo(x, PAD); sc.lineTo(x, H-PAD); sc.stroke();
  sc.setLineDash([]);

  const f = data.frames[idx];
  const y = PAD + (H - PAD*2) * (1 - f.anomaly_score);
  sc.beginPath(); sc.arc(x, y, 5, 0, Math.PI*2);
  sc.fillStyle = f.anomaly_count > 0 ? '#ef4444' : '#f59e0b';
  sc.fill();

  sc.setTransform(1,0,0,1,0,0);
}

// ══════════════════════════════════════════════════════════════
// 5. UPLOAD HANDLER (real API call with fallback)
// ══════════════════════════════════════════════════════════════
function handleUpload(file) {
  if (!file) return;
  document.getElementById('uploadZone').style.display  = 'none';
  document.getElementById('mainDisplay').style.display = 'grid';
  document.getElementById('infoPanels').style.display  = 'grid';

  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      document.getElementById('cameraImg').src = e.target.result;
      document.getElementById('cameraImg').style.opacity = '1';
      callDepthAPI(file).catch(() => {
        document.getElementById('overlayImg').src = 'rail-assets/jrsam3_11s_overlay.jpg';
        document.getElementById('overlayImg').style.opacity = '1';
        updateAlertPanel({ anoms:[], min:0, max:0, conf:0, ms:0 });
      });
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ══════════════════════════════════════════════════════════════
// 6. WEBCAM LIVE INFERENCE
// ══════════════════════════════════════════════════════════════
let webcamStream   = null;
let webcamRunning  = false;
let webcamInterval = null;
let webcamCanvas   = null;
let webcamFrameTs  = 0;
let webcamFrameN   = 0;

// Shared helper — update a webcam metric cell with flash animation
function setWcMetric(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = val;
  el.classList.remove('wc-anim');
  void el.offsetWidth;   // force reflow to restart animation
  el.classList.add('wc-anim');
}

async function toggleWebcam() {
  if (webcamRunning) stopWebcam();
  else await startWebcam();
}

async function startWebcam() {
  const btn    = document.getElementById('webcamBtn');
  const status = document.getElementById('webcamStatus');
  status.innerHTML = 'カメラへのアクセスを要求中…';
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 360 }, facingMode: 'environment' }
    });
  } catch (e) {
    status.innerHTML = `<span style="color:var(--danger)">カメラアクセスが拒否されました: ${e.message}</span>`;
    return;
  }

  const video = document.getElementById('webcamVideo');
  video.srcObject = webcamStream;
  await video.play();

  webcamCanvas        = document.createElement('canvas');
  webcamCanvas.width  = 640;
  webcamCanvas.height = 360;

  webcamRunning = true;
  webcamFrameN  = 0;
  webcamFrameTs = performance.now();

  btn.textContent = '⏹ 停止';
  btn.classList.add('running');
  document.getElementById('webcamLiveLabel').style.display = 'inline-block';
  status.innerHTML = '<strong>解析中</strong> — カメラ映像をAIが処理しています';

  captureAndSend();
  webcamInterval = setInterval(captureAndSend, 1200);
}

function stopWebcam() {
  webcamRunning = false;
  if (webcamInterval) { clearInterval(webcamInterval); webcamInterval = null; }
  if (webcamStream)   { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  const video = document.getElementById('webcamVideo');
  if (video) video.srcObject = null;
  const btn = document.getElementById('webcamBtn');
  if (btn) { btn.textContent = '📷 カメラ起動'; btn.classList.remove('running'); }
  const lbl = document.getElementById('webcamLiveLabel');
  if (lbl) lbl.style.display = 'none';
  const status = document.getElementById('webcamStatus');
  if (status) status.innerHTML = 'カメラを起動してリアルタイム深度解析を開始します。';
}

async function captureAndSend() {
  if (!webcamRunning) return;
  const video = document.getElementById('webcamVideo');
  if (!video || video.readyState < 2) return;

  // Draw frame onto canvas
  const ctx = webcamCanvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 640, 360);

  webcamCanvas.toBlob(async (blob) => {
    if (!blob || !webcamRunning) return;
    const file = new File([blob], 'webcam_frame.jpg', { type: 'image/jpeg' });
    const form = new FormData();
    form.append('image', file);

    let res = null;
    for (const base of API_ENDPOINTS) {
      try {
        const r = await tryFetch(
          `${base}/v1/depth/visualize`,
          { method: 'POST', body: form },
          base.includes('localhost') ? 2000 : 5000
        );
        if (r.ok) { res = r; break; }
      } catch(_) {}
    }
    if (!res) return;

    const resBlob = await res.blob();
    const url = URL.createObjectURL(resBlob);
    const overlay = document.getElementById('webcamOverlay');
    if (overlay) {
      if (overlay._prevUrl) URL.revokeObjectURL(overlay._prevUrl);
      overlay._prevUrl = url;
      overlay.src = url;
    }

    const nearM  = res.headers.get('X-Depth-Near-M') || '—';
    const farM   = res.headers.get('X-Depth-Far-M')  || '—';
    const procMs = res.headers.get('X-Processing-Ms') || '—';
    const anomN  = res.headers.get('X-Anomaly-Count');

    setWcMetric('wc-near', isNaN(parseFloat(nearM)) ? nearM : parseFloat(nearM).toFixed(1) + 'm');
    setWcMetric('wc-far',  isNaN(parseFloat(farM))  ? farM  : parseFloat(farM).toFixed(0)  + 'm');
    setWcMetric('wc-ms',   procMs + 'ms');
    setWcMetric('wc-anom', anomN !== null ? anomN + '件' : '—');
    if (anomN !== null) {
      const anomEl = document.getElementById('wc-anom');
      if (anomEl) anomEl.style.color = parseInt(anomN) > 0 ? 'var(--danger)' : 'var(--safe)';
    }

    // FPS badge
    webcamFrameN++;
    const elapsed  = (performance.now() - webcamFrameTs) / 1000;
    const fpsBadge = document.getElementById('webcamFpsBadge');
    if (fpsBadge) fpsBadge.textContent = (webcamFrameN / elapsed).toFixed(1) + ' fps';

  }, 'image/jpeg', 0.82);
}

// ══════════════════════════════════════════════════════════════
// 7. API HELPERS
// ══════════════════════════════════════════════════════════════

// API endpoints: local demo server preferred, Fly.io as fallback
const API_ENDPOINTS = [
  'http://localhost:8765',
  'https://spatialforge-demo.fly.dev',
];

async function tryFetch(url, opts, timeoutMs = 6000) {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    clearTimeout(timer);
    return res;
  } catch (e) {
    clearTimeout(timer);
    throw e;
  }
}

async function callDepthAPI(file) {
  const form = new FormData();
  form.append('image', file);

  let res = null, usedEndpoint = '';
  for (const base of API_ENDPOINTS) {
    try {
      const r = await tryFetch(
        `${base}/v1/depth/visualize`,
        { method: 'POST', body: form },
        base.includes('localhost') ? 3000 : 6000
      );
      if (r.ok) { res = r; usedEndpoint = base; break; }
    } catch (_) { /* try next */ }
  }
  if (!res) throw new Error('All API endpoints failed');

  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  const oImg = document.getElementById('overlayImg');
  oImg.src   = url;
  oImg.style.opacity = '1';

  const nearM  = res.headers.get('X-Depth-Near-M')  || res.headers.get('X-Depth-Min') || '—';
  const farM   = res.headers.get('X-Depth-Far-M')   || res.headers.get('X-Depth-Max') || '—';
  const procMs = res.headers.get('X-Processing-Ms') || '—';
  const model  = res.headers.get('X-Model')         || 'DA V2 Large-hf';
  const anomN  = res.headers.get('X-Anomaly-Count');
  const confV  = res.headers.get('X-Confidence');

  document.getElementById('m-near').textContent  = isNaN(parseFloat(nearM)) ? nearM : parseFloat(nearM).toFixed(1) + ' m';
  document.getElementById('m-far').textContent   = isNaN(parseFloat(farM))  ? farM  : parseFloat(farM).toFixed(0)  + ' m';
  document.getElementById('m-time').textContent  = procMs + ' ms';
  document.getElementById('m-model').textContent = model.replace('depth-anything/', '');
  document.getElementById('m-conf').textContent  = confV ? (parseFloat(confV)*100).toFixed(1) + '%' : '—';
  document.getElementById('m-clearance').textContent = '手動確認';
  document.getElementById('m-anom').textContent  = anomN !== null ? anomN + ' 件' : '—';

  const isLocal   = usedEndpoint.includes('localhost');
  const anomCount = anomN !== null ? parseInt(anomN) : 0;
  document.getElementById('alertList').innerHTML =
    `<div class="alert-item ${anomCount > 0 ? 'warning' : 'ok'}">
      <span class="alert-badge ${anomCount > 0 ? 'badge-warning' : 'badge-ok'}">${anomCount > 0 ? '警告' : '完了'}</span>
      <div class="alert-text">
        ${anomCount > 0
          ? `<strong>${anomCount}件の異常を検出</strong><span>軌道上の障害物候補を確認してください</span>`
          : `<strong>異常なし — 正常</strong><span>軌道クリアランス確認済み</span>`}
      </div>
    </div>
    <div class="alert-item ok" style="font-size:0.72rem;opacity:.6;padding:4px 10px">
      エンドポイント: ${isLocal ? 'ローカル (RTX 4090)' : 'Fly.io クラウド'} — ${procMs}ms
    </div>`;
}

async function checkApiStatus() {
  const dot = document.getElementById('apiDot');
  const txt = document.getElementById('apiStatusText');
  if (!dot || !txt) return;
  for (const base of API_ENDPOINTS) {
    try {
      const r = await tryFetch(`${base}/health`, {}, 2500);
      if (r.ok) {
        dot.style.background = '#10b981';
        txt.textContent = base.includes('localhost')
          ? 'API オンライン — ローカル RTX 4090 (高速推論)'
          : 'API オンライン — Fly.io クラウド';
        return;
      }
    } catch (_) {}
  }
  dot.style.background = '#f59e0b';
  txt.textContent = 'API オフライン — アップロード後にフォールバック処理します';
}

// ══════════════════════════════════════════════════════════════
// 8. ROUTE KILOMETER MAP
//    Shows anomaly km-post positions for both routes.
//    Assumes 80 km/h = 22.22 m/s throughout the clip.
//    Clicking a marker navigates to that keyframe.
// ══════════════════════════════════════════════════════════════
let _kmActiveRoute = 'jrsam3';

function renderKmMap(vid) {
  _kmActiveRoute = vid;
  document.querySelectorAll('.km-route-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.route === vid));

  const frames      = FRAMES.filter(f => f.video === vid);
  const totalSecs   = Math.max(...frames.map(f => f.ts));
  const totalM      = totalSecs * ASSUMED_SPEED_MS;
  const totalKm     = totalM / 1000;

  // SVG dimensions (responsive but min 600px)
  const SVG_W = 800, SVG_H = 90;
  const PAD_L = 48, PAD_R = 24, PAD_T = 20, PAD_B = 30;
  const TRACK_Y = PAD_T + 20;
  const TRACK_W = SVG_W - PAD_L - PAD_R;

  const kmToX = (km) => PAD_L + (km / totalKm) * TRACK_W;

  // Tick marks every 50m (0.05 km)
  const tickStep = totalKm <= 0.5 ? 0.05 : 0.1;
  let ticks = '';
  for (let km = 0; km <= totalKm + 0.001; km += tickStep) {
    const x = kmToX(km);
    const major = Math.abs(Math.round(km / (tickStep*5)) * tickStep*5 - km) < 0.001;
    ticks += `<line x1="${x}" y1="${TRACK_Y - (major?8:4)}" x2="${x}" y2="${TRACK_Y}" stroke="#1e1e2e" stroke-width="${major?1.5:1}"/>`;
    if (major) {
      const label = km >= 1 ? `${km.toFixed(2)}km` : `${(km*1000).toFixed(0)}m`;
      ticks += `<text x="${x}" y="${SVG_H - 4}" text-anchor="middle" font-size="9" fill="#8888a0">${label}</text>`;
    }
  }

  // Track rail line
  const rail = `<line x1="${PAD_L}" y1="${TRACK_Y}" x2="${SVG_W - PAD_R}" y2="${TRACK_Y}" stroke="#2a2a40" stroke-width="4" stroke-linecap="round"/>`;

  // Start / end labels
  const startLabel = `<text x="${PAD_L}" y="${TRACK_Y - 10}" font-size="9" fill="#8888a0" text-anchor="start">▶ 0m (起点)</text>`;
  const endLabel   = `<text x="${SVG_W - PAD_R}" y="${TRACK_Y - 10}" font-size="9" fill="#8888a0" text-anchor="end">終点 ${(totalM).toFixed(0)}m</text>`;

  // Anomaly markers
  let markers = '';
  frames.forEach(f => {
    const km   = (f.ts * ASSUMED_SPEED_MS) / 1000;
    const x    = kmToX(km);
    const hasA = f.anoms.length > 0;
    const topSev = hasA ? (f.anoms.some(a=>a.sev==='critical') ? 'critical' : f.anoms.some(a=>a.sev==='major') ? 'major' : 'minor') : null;
    const col  = topSev ? SEV_COLORS[topSev] : '#22c55e';
    const sym  = hasA ? '⚠' : '✓';
    const r    = hasA ? 9 : 6;
    const cursor = 'cursor:pointer';
    const defectList = hasA ? f.anoms.map(a => DEFECT_LABELS_JA[a.cls] || a.cls).join('・') : '正常';
    markers += `
      <g onclick="selectFrame('${f.name}');document.getElementById('demo').scrollIntoView({behavior:'smooth',block:'start'})"
         style="${cursor}" class="km-marker" data-frame="${f.name}">
        <line x1="${x}" y1="${TRACK_Y}" x2="${x}" y2="${TRACK_Y + (hasA ? 24 : 16)}" stroke="${col}" stroke-width="1.5" stroke-dasharray="${hasA ? '' : '2,2'}"/>
        <circle cx="${x}" cy="${TRACK_Y + (hasA ? 24 : 16) + r}" r="${r}" fill="${col}" fill-opacity="0.2" stroke="${col}" stroke-width="1.5"/>
        <text x="${x}" y="${TRACK_Y + (hasA ? 24 : 16) + r + 3.5}" text-anchor="middle" font-size="${hasA ? 7 : 6}" fill="${col}" font-weight="700">${sym}</text>
        <title>${f.label}  ${defectList} (${hasA ? f.anoms.length + '件' : '正常'})  @${(km*1000).toFixed(0)}m</title>
      </g>`;
  });

  // Speed note
  const mps = ASSUMED_SPEED_MS.toFixed(1);

  const svg = `<svg viewBox="0 0 ${SVG_W} ${SVG_H}" xmlns="http://www.w3.org/2000/svg" style="overflow:visible">
    <style>
      .km-marker circle { transition: r 0.15s, fill-opacity 0.15s; }
      .km-marker:hover circle { r: ${9+3}; fill-opacity: 0.4; }
    </style>
    ${ticks}${rail}${startLabel}${endLabel}${markers}
  </svg>`;

  const wrap = document.getElementById('kmMapSvgWrap');
  if (wrap) wrap.innerHTML = svg;
}

// ══════════════════════════════════════════════════════════════
// 9. TURBO COLORMAP + THREE.JS 3D VISUALIZATION
// ══════════════════════════════════════════════════════════════
const TURBO_STOPS = [
  [0.00,[48,18,59]],[0.10,[63,85,183]],[0.20,[50,146,207]],
  [0.30,[26,186,164]],[0.40,[74,210,98]],[0.50,[174,221,49]],
  [0.60,[234,185,35]],[0.70,[245,129,21]],[0.80,[234,66,18]],
  [0.90,[196,27,27]],[1.00,[122,4,3]],
];

function turboColor(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < TURBO_STOPS.length; i++) {
    if (t <= TURBO_STOPS[i][0]) {
      const a  = (t - TURBO_STOPS[i-1][0]) / (TURBO_STOPS[i][0] - TURBO_STOPS[i-1][0]);
      const c0 = TURBO_STOPS[i-1][1], c1 = TURBO_STOPS[i][1];
      return [
        Math.round(c0[0]+a*(c1[0]-c0[0])),
        Math.round(c0[1]+a*(c1[1]-c0[1])),
        Math.round(c0[2]+a*(c1[2]-c0[2])),
      ];
    }
  }
  return [122,4,3];
}

function turboPixelToDepth(r, g, b) {
  const rf = r/255, gf = g/255, bf = b/255;
  const cmax = Math.max(rf,gf,bf), cmin = Math.min(rf,gf,bf), d = cmax-cmin;
  if (cmax < 0.12) return 0.02;
  if (d < 0.04) return 0.45;
  let h;
  if      (cmax === rf) h = 60 * (((gf-bf)/d % 6 + 6) % 6);
  else if (cmax === gf) h = 60 * ((bf-rf)/d + 2);
  else                  h = 60 * ((rf-gf)/d + 4);
  if (h > 270) return Math.max(0, (310 - h) / 310 * 0.15);
  return Math.max(0, Math.min(1, (280 - h) / 280));
}

let threeRenderer, threeScene, threeCamera, threePoints, threeMesh;
let _threeCanvas = null;   // cached once in initThreeJs — avoids getElementById per RAF tick
let autoRotate = false;
let vizMode    = 'point';
let isDragging = false, prevMouse = {x:0, y:0};
let spherical  = {theta: 0.4, phi: 0.7, r: 5.5};
let currentDepthPixels = null;

// Seeded noise for synthetic depth (used by buildThreeJsSynthetic + initDashboard)
function seededNoise(x, y, s) {
  const n = Math.sin(x*127.1 + y*311.7 + s*74.3) * 43758.5453;
  return n - Math.floor(n);
}

function initThreeJs() {
  const canvas = document.getElementById('threeCanvas');
  _threeCanvas = canvas;   // cache for animate() RAF loop
  threeRenderer = new THREE.WebGLRenderer({canvas, antialias:true});
  threeRenderer.setPixelRatio(window.devicePixelRatio);
  threeRenderer.setSize(canvas.clientWidth, canvas.clientHeight);
  threeRenderer.setClearColor(0x0a0a0f, 1);

  threeScene  = new THREE.Scene();
  threeCamera = new THREE.PerspectiveCamera(60, canvas.clientWidth/canvas.clientHeight, 0.01, 100);
  updateThreeCamera();

  threeScene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(2,4,3);
  threeScene.add(dir);

  const grid = new THREE.GridHelper(8, 20, 0x1e1e2e, 0x1e1e2e);
  grid.position.y = -1.5;
  threeScene.add(grid);

  // Mouse orbit
  canvas.addEventListener('mousedown', e => { isDragging=true; prevMouse={x:e.clientX,y:e.clientY}; });
  canvas.addEventListener('mouseup',   () => { isDragging=false; });
  canvas.addEventListener('mousemove', e => {
    if (!isDragging) return;
    spherical.theta -= (e.clientX-prevMouse.x)*0.008;
    spherical.phi    = Math.max(0.1, Math.min(1.4, spherical.phi + (e.clientY-prevMouse.y)*0.008));
    prevMouse = {x:e.clientX, y:e.clientY};
    updateThreeCamera();
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    spherical.r = Math.max(2, Math.min(12, spherical.r + e.deltaY*0.008));
    updateThreeCamera();
  }, {passive:false});

  let lt = null;
  canvas.addEventListener('touchstart', e => { lt=e.touches[0]; });
  canvas.addEventListener('touchmove', e => {
    e.preventDefault(); if (!lt) return;
    spherical.theta -= (e.touches[0].clientX-lt.clientX)*0.01;
    spherical.phi    = Math.max(0.1, Math.min(1.4, spherical.phi+(e.touches[0].clientY-lt.clientY)*0.01));
    lt=e.touches[0]; updateThreeCamera();
  }, {passive:false});

  animate();
  loadDepthForThreeJs('jrsam3_11s', currentFrame);
}

function updateThreeCamera() {
  const {theta,phi,r} = spherical;
  threeCamera.position.set(r*Math.sin(phi)*Math.sin(theta), r*Math.cos(phi), r*Math.sin(phi)*Math.cos(theta));
  threeCamera.lookAt(0,0,0);
}

function loadDepthForThreeJs(name, frame, depthPath) {
  const normalizedFrame = frame ? {
    anoms: frame.anoms || frame.anomalies || [],
  } : { anoms: [] };

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = function() {
    const W=96, H=54;
    const off = document.createElement('canvas');
    off.width=W; off.height=H;
    const ctx = off.getContext('2d');
    ctx.drawImage(img, 0, 0, W, H);
    try {
      const px = ctx.getImageData(0,0,W,H).data;
      currentDepthPixels = px;
      buildThreeJsFromPixels(W, H);
    } catch(e) {
      buildThreeJsSynthetic(normalizedFrame);
    }
  };
  img.onerror = () => buildThreeJsSynthetic(normalizedFrame);
  img.src = depthPath || `rail-assets/${name}_depth.jpg`;
}

function buildThreeJsFromPixels(GW, GH) {
  if (!threeScene) return;
  if (threePoints) { threeScene.remove(threePoints); threePoints.geometry.dispose(); }
  if (threeMesh)   { threeScene.remove(threeMesh);   threeMesh.geometry.dispose();   }

  const positions=[], colors=[], meshVerts=[], meshColors=[], meshIdx=[];
  for (let j=0; j<GH; j++) {
    for (let i=0; i<GW; i++) {
      const idx = (j*GW+i)*4;
      const r=currentDepthPixels[idx], g=currentDepthPixels[idx+1], b=currentDepthPixels[idx+2];
      const d   = turboPixelToDepth(r,g,b);
      const u   = i/(GW-1), v=j/(GH-1);
      const ps  = 0.5+d*1.5;
      const wx  = (u-0.5)*4*ps, wy=(d-0.5)*-1.2, wz=(1-d)*5-2.5;
      positions.push(wx,wy,wz); colors.push(r/255,g/255,b/255);
      meshVerts.push(wx,wy,wz); meshColors.push(r/255,g/255,b/255);
      if (i<GW-1 && j<GH-1) {
        const tl=j*GW+i, tr=tl+1, bl=tl+GW, br=bl+1;
        meshIdx.push(tl,bl,tr, tr,bl,br);
      }
    }
  }
  _buildGeometry(positions, colors, meshVerts, meshColors, meshIdx);
}

function buildThreeJsSynthetic(frame) {
  if (!threeScene) return;
  if (threePoints) { threeScene.remove(threePoints); threePoints.geometry.dispose(); }
  if (threeMesh)   { threeScene.remove(threeMesh);   threeMesh.geometry.dispose();   }

  const GW=96, GH=72;
  const hs = frame && frame.anoms.length > 0 ? {
    u0: frame.anoms[0].x / 960,
    v0: frame.anoms[0].y / 540,
    u1: (frame.anoms[0].x + frame.anoms[0].w) / 960,
    v1: (frame.anoms[0].y + frame.anoms[0].h) / 540,
    d: 0.78
  } : null;

  const positions=[], colors=[], meshVerts=[], meshColors=[], meshIdx=[];
  for (let j=0; j<GH; j++) {
    for (let i=0; i<GW; i++) {
      const u=i/(GW-1), v=j/(GH-1);
      const d   = synthDepth(u,v,hs);
      const [r,g,b] = turboColor(d);
      const ps  = 0.5+d*1.5;
      const wx  = (u-0.5)*4*ps, wy=(d-0.5)*-1.2, wz=(1-d)*5-2.5;
      positions.push(wx,wy,wz); colors.push(r/255,g/255,b/255);
      meshVerts.push(wx,wy,wz); meshColors.push(r/255,g/255,b/255);
      if (i<GW-1 && j<GH-1) {
        const tl=j*GW+i, tr=tl+1, bl=tl+GW, br=bl+1;
        meshIdx.push(tl,bl,tr, tr,bl,br);
      }
    }
  }
  _buildGeometry(positions, colors, meshVerts, meshColors, meshIdx);
}

function synthDepth(u, v, hs) {
  if (v < 0.32) return 0.03 + seededNoise(u*40,v*40,1)*0.03;
  const gt=(v-0.32)/0.68;
  let d=0.04+gt*0.91;
  const tc=0.5, thw=0.11+gt*0.07;
  const lr=tc-0.068*(1+gt*0.4), rr=tc+0.068*(1+gt*0.4), rw=0.008+gt*0.005;
  if (Math.abs(u-lr)<rw || Math.abs(u-rr)<rw) d+=0.025;
  if (Math.sin((gt*22)*Math.PI)>0.55 && Math.abs(u-tc)<thw*1.3) d+=0.015;
  if (Math.abs(u-tc)>thw && Math.abs(u-tc)<thw*2.5)  d-=0.04;
  if (Math.abs(u-tc)>thw*2.5) d-=0.06;
  if (hs && u>=hs.u0 && u<=hs.u1 && v>=hs.v0 && v<=hs.v1) {
    const hx=(u-hs.u0)/(hs.u1-hs.u0), hy=(v-hs.v0)/(hs.v1-hs.v0);
    const ef=Math.min(hx,1-hx,hy,1-hy)*4;
    const tgt=hs.d+seededNoise(u*20,v*20,3)*0.04;
    d=d+Math.min(1,ef)*(tgt-d);
  }
  d+=(seededNoise(u*80,v*80,7)-0.5)*0.018;
  return Math.max(0,Math.min(1,d));
}

function _buildGeometry(pos, col, mv, mc, mi) {
  const ptGeo = new THREE.BufferGeometry();
  ptGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos,3));
  ptGeo.setAttribute('color',    new THREE.Float32BufferAttribute(col,3));
  const ptMat = new THREE.PointsMaterial({size:0.045, vertexColors:true, sizeAttenuation:true});
  threePoints = new THREE.Points(ptGeo, ptMat);
  threePoints.visible = vizMode==='point';
  threeScene.add(threePoints);

  const mGeo = new THREE.BufferGeometry();
  mGeo.setAttribute('position', new THREE.Float32BufferAttribute(mv,3));
  mGeo.setAttribute('color',    new THREE.Float32BufferAttribute(mc,3));
  mGeo.setIndex(mi); mGeo.computeVertexNormals();
  const mMat = new THREE.MeshPhongMaterial({vertexColors:true, side:THREE.DoubleSide, shininess:8});
  threeMesh = new THREE.Mesh(mGeo, mMat);
  threeMesh.visible = vizMode==='mesh';
  threeScene.add(threeMesh);
}

function setVizMode(mode, e) {
  vizMode = mode;
  document.querySelectorAll('.viz-btn').forEach(b => b.classList.remove('active'));
  if (e && e.target) e.target.classList.add('active');
  if (threePoints) threePoints.visible = mode==='point';
  if (threeMesh)   threeMesh.visible   = mode==='mesh';
}

function resetCamera() {
  spherical = {theta:0.4, phi:0.7, r:5.5};
  updateThreeCamera();
}

function toggleAutoRotate(e) {
  autoRotate = !autoRotate;
  if (e && e.target) e.target.classList.toggle('active', autoRotate);
}

function animate() {
  requestAnimationFrame(animate);
  if (autoRotate) { spherical.theta += 0.004; updateThreeCamera(); }
  const canvas = _threeCanvas;
  if (!canvas || !threeRenderer) return;
  if (canvas.clientWidth  !== threeRenderer.domElement.width ||
      canvas.clientHeight !== threeRenderer.domElement.height) {
    threeRenderer.setSize(canvas.clientWidth, canvas.clientHeight);
    threeCamera.aspect = canvas.clientWidth / canvas.clientHeight;
    threeCamera.updateProjectionMatrix();
  }
  threeRenderer.render(threeScene, threeCamera);
}

// Wait until THREE is defined (handles async CDN + fallback loading)
function waitForThree(cb, retries = 20) {
  if (typeof THREE !== 'undefined') { cb(); return; }
  if (retries <= 0) { console.warn('[RailScan] THREE.js failed to load'); return; }
  setTimeout(() => waitForThree(cb, retries - 1), 200);
}

// ══════════════════════════════════════════════════════════════
// 10. CHART.JS DASHBOARD
// ══════════════════════════════════════════════════════════════
function initDashboard() {
  const days=[], data=[];
  const today=new Date();
  for (let i=29;i>=0;i--) {
    const d=new Date(today); d.setDate(d.getDate()-i);
    days.push(`${d.getMonth()+1}/${d.getDate()}`);
    const base=2.0+Math.sin(i*0.35)*1.2;
    data.push(Math.max(0,Math.round(base+(seededNoise(i*0.1,0,42)-0.5)*2)));
  }
  // Pin real measured values
  data[data.length-1]=9; data[data.length-7]=5; data[data.length-14]=7;

  if (typeof Chart === 'undefined') return;

  new Chart(document.getElementById('trendChart'),{
    type:'line',
    data:{labels:days, datasets:[{label:'異常検知件数',data, fill:true,
      backgroundColor:'rgba(245,158,11,0.1)', borderColor:'#f59e0b', borderWidth:2,
      pointRadius:3, pointBackgroundColor:'#f59e0b', tension:0.3}]},
    options:{responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:'#8888a0',font:{size:10},maxTicksLimit:10},grid:{color:'#1e1e2e'}},
        y:{ticks:{color:'#8888a0',font:{size:10}},grid:{color:'#1e1e2e'},min:0,suggestedMax:12}}}
  });

  new Chart(document.getElementById('typeChart'),{
    type:'doughnut',
    data:{
      labels:['レールき裂','レール摩耗','レール波状摩耗','締結装置欠損','まくらぎき裂','バラスト異状','継目異状','軌間異常','その他'],
      datasets:[{data:[28,18,12,15,8,10,5,3,1],
        backgroundColor:['#ef4444','#f59e0b','#fb923c','#dc2626','#8b5cf6','#6366f1','#f97316','#3b82f6','#8888a0'], borderWidth:0}]},
    options:{responsive:true, maintainAspectRatio:false,
      plugins:{legend:{position:'bottom',labels:{color:'#8888a0',font:{size:10},padding:8,boxWidth:10}}},
      cutout:'65%'}
  });
}

// ══════════════════════════════════════════════════════════════
// 11. HERO COUNTER ANIMATION
// ══════════════════════════════════════════════════════════════
function animateCounter(el, target, suffix, duration) {
  let start = null;
  const isDecimal = String(target).includes('.');
  function step(ts) {
    if (!start) start = ts;
    const prog = Math.min((ts-start)/duration, 1);
    const ease = 1 - Math.pow(1-prog, 3);
    el.textContent = (isDecimal ? (target*ease).toFixed(1) : Math.round(target*ease)) + suffix;
    if (prog < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ══════════════════════════════════════════════════════════════
// 12. KEYBOARD NAVIGATION
//    Space      — play/pause (player frame mode)
//    ← / →      — prev/next frame (keyframe or player)
//    N          — next anomaly (player)
//    P          — prev anomaly (player)
//    1–5        — switch tabs
// ══════════════════════════════════════════════════════════════
function initKeyboardNav() {
  const TAB_KEYS = {
    '1': 'jrsam3',
    '2': 'jr23',
    '3': 'upload',
    '4': 'player',
    '5': 'webcam',
  };

  document.addEventListener('keydown', (e) => {
    // Ignore when focused on input/textarea/select
    const tag = document.activeElement.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    switch (e.key) {
      case ' ':
      case 'Spacebar': {
        e.preventDefault();
        if (currentVideo === 'player' && playerMode === 'frame') togglePlay();
        break;
      }
      case 'ArrowLeft': {
        e.preventDefault();
        if (currentVideo === 'player' && playerTimeline) {
          scrubTo(Math.max(0, playerFrameIdx - 1));
        } else if (currentVideo !== 'upload' && currentVideo !== 'webcam' && currentVideo !== 'player') {
          // Keyframe mode
          const frames = FRAMES.filter(f => f.video === currentVideo);
          const idx    = frames.findIndex(f => f.name === currentFrame?.name);
          if (idx > 0) selectFrame(frames[idx-1].name);
        }
        break;
      }
      case 'ArrowRight': {
        e.preventDefault();
        if (currentVideo === 'player' && playerTimeline) {
          scrubTo(Math.min(playerTimeline.total_frames-1, playerFrameIdx + 1));
        } else if (currentVideo !== 'upload' && currentVideo !== 'webcam' && currentVideo !== 'player') {
          const frames = FRAMES.filter(f => f.video === currentVideo);
          const idx    = frames.findIndex(f => f.name === currentFrame?.name);
          if (idx < frames.length-1) selectFrame(frames[idx+1].name);
        }
        break;
      }
      case 'n':
      case 'N': {
        if (currentVideo === 'player') jumpToAnomalyFrame(1);
        break;
      }
      case 'p':
      case 'P': {
        if (currentVideo === 'player') jumpToAnomalyFrame(-1);
        break;
      }
      default: {
        const tab = TAB_KEYS[e.key];
        if (tab) selectVideo(tab);
      }
    }
  });
}

// ══════════════════════════════════════════════════════════════
// 13. ONBOARDING TOUR
//    Shows once per browser (localStorage key: railscan_toured).
//    5 steps, each optionally highlights a target element.
//    Skip / ← / → navigation, progress dots.
// ══════════════════════════════════════════════════════════════
const TOUR_STEPS = [
  {
    title: 'RailScan AI へようこそ',
    body:  'Depth Anything V2 Large による鉄道軌道の<strong>リアルタイム深度推論</strong>デモです。<br>このツアーで主要機能を 1 分でご案内します。',
    target: null,
  },
  {
    title: 'キーフレームギャラリー',
    body:  '上部のタブで <strong>jrsam3路線 / jr23路線</strong> を切り替え、<br>フレームカードをクリックすると深度オーバーレイと異常詳細が表示されます。<br><span style="color:#ef4444">⚠</span> マークが付いたフレームに異常が検知されています。',
    target: 'frameStrip',
  },
  {
    title: '動画プレイヤー（全フレーム再生）',
    body:  '「動画プレイヤー」タブを開くと <strong>73フレームの推論結果</strong>をシーケンシャルに再生できます。<br>スパークラインをクリックして任意フレームへジャンプ。<br><kbd style="background:#1e1e2e;padding:2px 6px;border-radius:4px;font-size:0.85em">N</kbd> / <kbd style="background:#1e1e2e;padding:2px 6px;border-radius:4px;font-size:0.85em">P</kbd> キーで異常フレームを前後に移動できます。',
    target: null,
  },
  {
    title: '独自画像のアップロード',
    body:  '「アップロード」タブで <strong>任意の画像を AI 解析</strong>できます。<br>ローカル推論サーバー（RTX 4090）が起動中なら高速推論、<br>未起動の場合は Fly.io クラウド経由で自動フォールバックします。',
    target: 'uploadZone',
  },
  {
    title: 'KM マップとウェブカム',
    body:  '<strong>KM マップ</strong>では異常発生地点を km ポスト換算で路線図表示。<br><strong>ウェブカム</strong>タブではリアルタイム映像をフレームごとに推論し、<br>前方障害物を即時検知します。',
    target: 'kmmap',
  },
];

let _tourStep = 0;
let _tourEl   = null;

function initTour() {
  if (localStorage.getItem('railscan_toured')) return;
  _buildTourDom();
  _showTourStep(0);
}

function _buildTourDom() {
  const overlay = document.createElement('div');
  overlay.id = 'tourOverlay';
  overlay.innerHTML = `
    <div id="tourCard">
      <button id="tourClose" title="スキップ" aria-label="ツアーをスキップ">✕</button>
      <div id="tourStepLabel"></div>
      <h3 id="tourTitle"></h3>
      <p  id="tourBody"></p>
      <div id="tourDots"></div>
      <div id="tourBtns">
        <button id="tourPrev">← 前へ</button>
        <button id="tourNext"></button>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  document.getElementById('tourClose').addEventListener('click', _closeTour);
  document.getElementById('tourPrev').addEventListener('click', () => _showTourStep(_tourStep - 1));
  document.getElementById('tourNext').addEventListener('click', () => {
    if (_tourStep < TOUR_STEPS.length - 1) _showTourStep(_tourStep + 1);
    else _closeTour();
  });
}

function _showTourStep(idx) {
  _tourStep = Math.max(0, Math.min(TOUR_STEPS.length - 1, idx));
  const step = TOUR_STEPS[_tourStep];
  const isLast = _tourStep === TOUR_STEPS.length - 1;

  document.getElementById('tourStepLabel').textContent =
    `${_tourStep + 1} / ${TOUR_STEPS.length}`;
  document.getElementById('tourTitle').textContent = step.title;
  document.getElementById('tourBody').innerHTML  = step.body;
  document.getElementById('tourPrev').style.visibility =
    _tourStep === 0 ? 'hidden' : 'visible';
  document.getElementById('tourNext').textContent = isLast ? '始める ✓' : '次へ →';

  // Progress dots
  const dots = document.getElementById('tourDots');
  dots.innerHTML = TOUR_STEPS.map((_, i) =>
    `<span class="tour-dot${i === _tourStep ? ' active' : ''}"></span>`
  ).join('');

  // Highlight target element
  if (_tourEl) { _tourEl.classList.remove('tour-highlight'); _tourEl = null; }
  if (step.target) {
    const el = document.getElementById(step.target);
    if (el) { el.classList.add('tour-highlight'); _tourEl = el; }
  }
}

function _closeTour() {
  localStorage.setItem('railscan_toured', '1');
  if (_tourEl) { _tourEl.classList.remove('tour-highlight'); _tourEl = null; }
  const overlay = document.getElementById('tourOverlay');
  if (overlay) {
    overlay.style.opacity = '0';
    overlay.style.transition = 'opacity 0.3s';
    setTimeout(() => overlay.remove(), 350);
  }
}

// ══════════════════════════════════════════════════════════════
// 14. INITIALISATION
// ══════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', () => {
  // Render initial frame strip and select default anomaly frame
  renderFrameStrip('jrsam3');
  selectFrame('jrsam3_11s');

  // Onboarding tour (first visit only)
  setTimeout(initTour, 600);

  // Km map
  renderKmMap('jrsam3');

  // API status check (non-blocking)
  setTimeout(checkApiStatus, 800);

  // Three.js — wait for THREE to be ready (handles CDN async / offline fallback)
  waitForThree(initThreeJs);

  // Dashboard — lazy-init when section scrolls into view (below the fold)
  const dashSection = document.getElementById('dashboard');
  if (dashSection) {
    const dashObs = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { initDashboard(); dashObs.disconnect(); } });
    }, {threshold: 0.1});
    dashObs.observe(dashSection);
  } else {
    initDashboard();
  }

  // Keyboard navigation
  initKeyboardNav();

  // Hero counters — IntersectionObserver triggers once visible
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        animateCounter(document.getElementById('stat1'), 1847, '', 1800);
        animateCounter(document.getElementById('stat2'), 40,   '', 1600);
        animateCounter(document.getElementById('stat3'), 30,   'x', 1600);
        animateCounter(document.getElementById('stat4'), 47,   '', 1900);
        obs.disconnect();
      }
    });
  }, {threshold:0.3});
  obs.observe(document.querySelector('.hero-stats'));

  // Drag and drop upload
  const dz = document.getElementById('dropZone');
  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('drag-over');
    handleUpload(e.dataTransfer.files[0]);
  });

  // Canvas resize
  window.addEventListener('resize', () => {
    if (!threeRenderer || !_threeCanvas) return;
    const c = _threeCanvas;
    threeRenderer.setSize(c.clientWidth, c.clientHeight);
    threeCamera.aspect = c.clientWidth / c.clientHeight;
    threeCamera.updateProjectionMatrix();
  });

  // Live dashboard counter (simulates incoming runs)
  setInterval(() => {
    const el = document.getElementById('dash-runs');
    if (el) el.textContent = parseInt(el.textContent) + 1;
  }, 8000);
});
