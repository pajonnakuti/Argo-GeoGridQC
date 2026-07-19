const API = "";
const MAP_W = 800, MAP_H = 720, PAD = 40;

let meta = null;
let gridCells = [];
let mapMode = "observations"; // observations | analysis
let gridMetric = "n_obs";
let selectedGrid = null;
let profilesCache = {};
let highlightGridId = null;
let gridStatus = {}; // grid_id -> {grid_status, pct_temp_match, ...}

function project(lon, lat) {
  const x = PAD + ((lon - meta.bounds.lon_min) / (meta.bounds.lon_max - meta.bounds.lon_min)) * (MAP_W - 2 * PAD);
  const y = PAD + ((meta.bounds.lat_max - lat) / (meta.bounds.lat_max - meta.bounds.lat_min)) * (MAP_H - 2 * PAD);
  return { x, y };
}

function metricRange(metric) {
  const vals = gridCells.filter((c) => c.has_data).map((c) => c[metric] ?? c.metric_value);
  if (!vals.length) return [0, 1];
  return [Math.min(...vals), Math.max(...vals)];
}

function lerpColor(t) {
  const r = Math.round(8 + (0 - 8) * t);
  const g = Math.round(24 + (245 - 24) * t);
  const b = Math.round(32 + (212 - 32) * t);
  return `rgb(${r},${g},${b})`;
}

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

document.getElementById("diveBtn").addEventListener("click", () => {
  document.getElementById("login-scene").classList.add("submerging");
  setTimeout(() => {
    document.getElementById("login-scene").classList.add("hidden");
    document.getElementById("dash-scene").classList.remove("hidden");
    initApp();
  }, 1100);
});

document.querySelectorAll(".nav-station").forEach((el) => {
  el.addEventListener("click", () => {
    document.querySelectorAll(".nav-station").forEach((n) => n.classList.remove("active"));
    el.classList.add("active");
    renderPage(el.dataset.page);
  });
});

async function initApp() {
  meta = await api("/api/meta");
  const [data, statusRes] = await Promise.all([
    api("/api/grids?metric=n_obs"),
    api("/api/grids/status").catch(() => ({ available: false, grids: {} })),
  ]);
  gridCells = data.cells;
  if (statusRes.available) gridStatus = statusRes.grids;
  const pill = document.getElementById("statusPill");
  if (pill) {
    if (meta.warming) {
      pill.innerHTML = '<span class="pulse-dot"></span> Warming cache — map may update shortly…';
      const poll = setInterval(async () => {
        try {
          meta = await api("/api/meta");
          if (!meta.warming) {
            gridCells = (await api("/api/grids?metric=n_obs")).cells;
            pill.innerHTML = statusText();
            paintMap();
            clearInterval(poll);
          }
        } catch { /* retry */ }
      }, 8000);
    } else {
      pill.innerHTML = statusText();
    }
  }
  renderPage("grid");
}

function statusText() {
  const rf = meta.rf_models_loaded ? ` · RF ${meta.rf_models_loaded}/2` : "";
  return `<span class="pulse-dot"></span> ALL_REGIONS_UNIFIED · ${meta.populated_grids} grids · ${(meta.rows / 1e6).toFixed(1)}M obs${rf}`;
}

function renderPage(page) {
  const c = document.getElementById("pageContent");
  if (page === "grid") {
    c.innerHTML = gridPageHtml();
    wireGridPage();
    paintMap();
    return;
  }
  if (page === "qc") {
    c.innerHTML = qcPageHtml();
    wireQcPage();
  }
}

function gridPageHtml() {
  const b = meta.bounds;
  return `
    <div class="page-head">
      <div class="eyebrow">Indian Ocean · 5° Mesoscale Lattice</div>
      <h2>Mesoscale Hydrographic Observing Grid (MHOG)</h2>
      <p>Interactive observatory over <b>ALL_REGIONS_UNIFIED.parquet</b> — ${meta.total_grids} cells · ${b.lon_min}°–${b.lon_max}°E · ${b.lat_min}°–${b.lat_max}°N · ${meta.grid_rows}×${meta.cols} · <b>${meta.populated_grids}</b> populated · ${(meta.rows / 1e6).toFixed(1)}M profiles (Argo · CTD · XBT)</p>
    </div>
    <div class="grid-toolbar">
      <button class="vt-btn active" data-mode="observations">Observations</button>
      <button class="vt-btn" data-mode="analysis">Analysis</button>
      <select id="metricSelect" style="display:none">
        <option value="n_obs">Observation density</option>
        <option value="mean_temp">Mean temperature</option>
        <option value="mean_psal">Mean salinity</option>
        <option value="mean_depth">Mean depth</option>
      </select>
      <div id="instToggles" class="inst-toggles">
        <label><input type="checkbox" data-t="ARGO" checked> Argo</label>
        <label><input type="checkbox" data-t="CTD" checked> CTD</label>
        <label><input type="checkbox" data-t="XBT" checked> XBT</label>
      </div>
      <div class="grid-search">
        <input id="gridSearch" type="number" min="1" max="400" placeholder="Grid ID…">
        <button class="vt-btn" id="gridSearchBtn">Go</button>
      </div>
      <button class="vt-btn" id="resetViewBtn">Reset map</button>
    </div>
    <div class="grid-map-shell">
      <div>
        <div class="grid-map-wrap" id="mapWrap">
          <img class="basemap" src="/static/basemap.png" alt="Indian Ocean basemap" onerror="this.style.display='none'">
          <svg id="gridMapSvg" viewBox="0 0 ${MAP_W} ${MAP_H}" preserveAspectRatio="xMidYMid meet"></svg>
          <div class="map-tooltip" id="mapTooltip"></div>
        </div>
        <div class="legend-scale" id="legendScale" style="display:none"><span>Low</span><div class="bar"></div><span>High</span></div>
        <div class="inst-legend">
          <span><i class="dot argo"></i> Argo</span>
          <span><i class="dot ctd"></i> CTD</span>
          <span><i class="dot xbt"></i> XBT</span>
          <span><i class="dot fail"></i> QC Failed/Partial</span>
        </div>
        <div class="heatmap-panel" id="heatmapPanel" style="display:none">
          <div class="gd-section-title">Profile density · Gaussian-smoothed</div>
          <canvas class="heatmap-canvas" id="heatmapCanvas" width="640" height="140"></canvas>
        </div>
      </div>
      <div class="grid-detail-panel" id="detailPanel">
        <div class="gd-empty">Select a mode — <b>Observations</b> zooms to profile locations; <b>Analysis</b> shows thermohaline &amp; QC diagnostics.</div>
      </div>
    </div>`;
}

function wireGridPage() {
  document.querySelectorAll(".vt-btn[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".vt-btn[data-mode]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      mapMode = btn.dataset.mode;
      document.getElementById("metricSelect").style.display = mapMode === "analysis" ? "" : "none";
      document.getElementById("legendScale").style.display = mapMode === "analysis" ? "flex" : "none";
      document.getElementById("instToggles").style.display = mapMode === "observations" ? "flex" : "none";
      resetMapView();
      paintMap();
      document.getElementById("detailPanel").innerHTML =
        `<div class="gd-empty">${mapMode === "observations" ? "Click a grid to zoom and inspect Argo / CTD / XBT observations." : "Click a grid for thermohaline statistics, raw QC flags, and model-predicted QC comparison."}</div>`;
    });
  });

  document.getElementById("metricSelect").addEventListener("change", (e) => {
    gridMetric = e.target.value;
    paintMap();
  });
  document.getElementById("resetViewBtn").addEventListener("click", resetMapView);
  document.getElementById("gridSearchBtn").addEventListener("click", () => {
    const gid = parseInt(document.getElementById("gridSearch").value, 10);
    if (gid > 0) selectGrid(gid);
  });
  document.getElementById("gridSearch").addEventListener("keydown", (e) => {
    if (e.key === "Enter") document.getElementById("gridSearchBtn").click();
  });
  document.querySelectorAll("#instToggles input").forEach((cb) => {
    cb.addEventListener("change", () => { if (selectedGrid) paintPoints(); });
  });

  const svg = document.getElementById("gridMapSvg");
  const tip = document.getElementById("mapTooltip");
  const wrap = document.getElementById("mapWrap");

  svg.addEventListener("mousemove", (e) => {
    const t = e.target.closest(".gcell.ocean");
    if (!t) { tip.classList.remove("show"); return; }
    const id = t.dataset.id;
    const cell = gridCells.find((c) => String(c.grid_id) === id);
    const st = gridStatus[parseInt(id, 10)];
    const rect = wrap.getBoundingClientRect();
    tip.style.left = e.clientX - rect.left + 12 + "px";
    tip.style.top = e.clientY - rect.top + 12 + "px";
    tip.innerHTML = `<b>GRID-${id}</b>${st ? `<br>QC: ${st.grid_status}` : ""}<br>${cell.lat_min}°–${cell.lat_max}°N, ${cell.lon_min}°–${cell.lon_max}°E<br>${(cell.n_obs || 0).toLocaleString()} observations`;
    tip.classList.add("show");
  });
  svg.addEventListener("mouseleave", () => tip.classList.remove("show"));
  svg.addEventListener("click", (e) => {
    const t = e.target.closest(".gcell.ocean");
    if (!t || !t.dataset.hasData) return;
    selectGrid(parseInt(t.dataset.id, 10));
  });
}

function resetMapView() {
  selectedGrid = null;
  highlightGridId = null;
  document.getElementById("mapWrap").classList.remove("focus-active");
  document.getElementById("gridMapSvg").setAttribute("viewBox", `0 0 ${MAP_W} ${MAP_H}`);
  document.getElementById("heatmapPanel").style.display = "none";
  paintMap();
}

function paintMap() {
  const svg = document.getElementById("gridMapSvg");
  if (!svg) return;
  const [lo, hi] = metricRange(gridMetric);
  let cells = "";
  for (const c of gridCells) {
    const p0 = project(c.lon_min, c.lat_max);
    const p1 = project(c.lon_max, c.lat_min);
    const w = p1.x - p0.x, h = p1.y - p0.y;
    let fill = "none", op = 0;
    if (c.has_data) {
      if (mapMode === "analysis") {
        const v = c[gridMetric] ?? 0;
        const t = hi > lo ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0;
        fill = lerpColor(t);
        op = 0.55;
      } else {
        fill = "rgba(8,24,32,0.4)";
        op = 0.35;
      }
    }
    const cls = c.has_data ? "ocean" : "land";
    const st = gridStatus[c.grid_id];
    const problem = st && (st.grid_status === "FAILED" || st.grid_status === "PARTIAL");
    const hl = highlightGridId === c.grid_id ? " highlighted" : "";
    const sel = selectedGrid === c.grid_id ? " selected focused" : "";
    const prob = problem ? " problem-grid" : "";
    cells += `<rect class="gcell ${cls}${hl}${sel}${prob}" data-id="${c.grid_id}" data-has-data="${c.has_data ? 1 : ""}" data-status="${st?.grid_status || ""}" x="${p0.x}" y="${p0.y}" width="${w}" height="${h}" fill="${fill}" fill-opacity="${op}" stroke="${problem ? "#e5654e" : "rgba(127,255,212,0.25)"}" stroke-width="${problem ? 2 : 0.5}"/>`;
    if (c.has_data) {
      cells += `<text class="gcell-id" x="${p0.x + w / 2}" y="${p0.y + h / 2}" text-anchor="middle" dominant-baseline="middle" font-size="5" fill="rgba(234,246,246,0.55)">${c.grid_id}</text>`;
    }
  }
  svg.innerHTML = cells + '<g id="pointsLayer"></g>';
  if (mapMode === "observations" && selectedGrid) paintPoints();
}

async function paintPoints() {
  const layer = document.getElementById("pointsLayer");
  if (!layer || !selectedGrid) return;
  const visible = {};
  document.querySelectorAll("#instToggles input").forEach((cb) => { visible[cb.dataset.t] = cb.checked; });
  const c = gridCells.find((g) => g.grid_id === selectedGrid);
  if (!c) return;
  if (!profilesCache[selectedGrid]) {
    try {
      const res = await api(`/api/grids/${selectedGrid}/profiles?limit=200`);
      profilesCache[selectedGrid] = res.profiles;
    } catch { profilesCache[selectedGrid] = []; }
  }
  let dots = "";
  for (const p of profilesCache[selectedGrid]) {
    const inst = (p.instrument || "ARGO").toUpperCase();
    if (!visible[inst]) continue;
    const col = inst === "CTD" ? "#ffd166" : inst === "XBT" ? "#ef476f" : "#00f5d4";
    const pt = project(p.lon, p.lat);
    dots += `<circle class="gpoint" cx="${pt.x}" cy="${pt.y}" r="1.8" fill="${col}" opacity="0.9"/>`;
  }
  layer.innerHTML = dots;
}

function zoomToGrid(gid) {
  const cell = gridCells.find((c) => c.grid_id === gid);
  const wrap = document.getElementById("mapWrap");
  const svg = document.getElementById("gridMapSvg");
  if (!cell || !wrap || !svg) return;
  wrap.classList.add("focus-active");
  const p0 = project(cell.lon_min, cell.lat_max);
  const p1 = project(cell.lon_max, cell.lat_min);
  const pad = 35;
  svg.setAttribute("viewBox", `${p0.x - pad} ${p0.y - pad} ${p1.x - p0.x + pad * 2} ${p1.y - p0.y + pad * 2}`);
}

async function selectGrid(gid) {
  selectedGrid = gid;
  document.getElementById("gridSearch").value = gid;
  paintMap();

  if (mapMode === "observations") {
    zoomToGrid(gid);
    await paintPoints();
    const panel = document.getElementById("detailPanel");
    panel.innerHTML = '<div class="loading">Loading observations…</div>';
    try {
      const [detail, heat] = await Promise.all([
        api(`/api/grids/${gid}`),
        api(`/api/grids/${gid}/heatmap`),
      ]);
      renderObservationsPanel(panel, detail);
      renderHeatmap(heat);
      document.getElementById("heatmapPanel").style.display = "block";
    } catch (e) {
      panel.innerHTML = `<div class="gd-empty">Error: ${e.message}</div>`;
    }
  } else {
    document.getElementById("mapWrap").classList.remove("focus-active");
    document.getElementById("gridMapSvg").setAttribute("viewBox", `0 0 ${MAP_W} ${MAP_H}`);
    document.getElementById("heatmapPanel").style.display = "none";
    const panel = document.getElementById("detailPanel");
    panel.innerHTML = '<div class="loading">Running grid analysis…</div>';
    try {
      const data = await api(`/api/grids/${gid}/analysis`);
      renderAnalysisPanel(panel, data);
    } catch (e) {
      panel.innerHTML = `<div class="gd-empty">Error: ${e.message}</div>`;
    }
  }

  if (highlightGridId === gid) {
    document.querySelectorAll(".gcell").forEach((el) => {
      if (parseInt(el.dataset.id, 10) === gid) el.classList.add("highlighted");
    });
  }
}

function instCountsHtml(instruments) {
  const colors = { ARGO: "#00f5d4", CTD: "#ffd166", XBT: "#ef476f" };
  const total = (instruments || []).reduce((s, x) => s + x.n, 0) || 1;
  let pie = "", leg = "", off = 0;
  (instruments || []).forEach((x) => {
    const pct = (x.n / total) * 100;
    leg += `<span><span class="swatch" style="background:${colors[x.instrument] || '#aaa'}"></span>${x.instrument} <b>${x.n.toLocaleString()}</b> (${pct.toFixed(0)}%)</span>`;
    pie += `${colors[x.instrument] || '#aaa'} ${off}% ${off + pct}%`;
    off += pct;
  });
  return { pie, leg };
}

function qcPieHtml(qc, title) {
  const colors = { Good: "#4fd69c", "Probably good": "#e8b84b", Bad: "#e5654e", Missing: "#8592a0" };
  const total = (qc || []).reduce((s, x) => s + x.n, 0) || 1;
  let pie = "", leg = "", off = 0;
  (qc || []).forEach((x) => {
    const pct = (x.n / total) * 100;
    const col = colors[x.label] || "#aaa";
    leg += `<span><span class="swatch" style="background:${col}"></span>${x.label} <b>${x.n.toLocaleString()}</b> (${pct.toFixed(0)}%)</span>`;
    pie += `${col} ${off}% ${off + pct}%`;
    off += pct;
  });
  return `<div class="gd-section-title">${title}</div>
    <div class="pie-row"><div class="pie-ring" style="background:conic-gradient(${pie || '#333 0 100%'})"></div>
    <div class="pie-legend">${leg || "No flags"}</div></div>`;
}

function renderObservationsPanel(panel, d) {
  const bb = d.bbox;
  const { pie, leg } = instCountsHtml(d.instruments);
  const argo = d.instruments?.find((x) => x.instrument === "ARGO")?.n || 0;
  const ctd = d.instruments?.find((x) => x.instrument === "CTD")?.n || 0;
  const xbt = d.instruments?.find((x) => x.instrument === "XBT")?.n || 0;

  panel.innerHTML = `
    <div class="gd-header"><span class="gid">GRID-${d.grid_id}</span><span class="mode-tag obs">Observations</span></div>
    <div class="sub-bounds">${bb.lat_min}°–${bb.lat_max}°N · ${bb.lon_min}°–${bb.lon_max}°E · row ${bb.row} col ${bb.col}</div>
    <div class="inst-badges">
      <span class="badge argo">Argo ${argo.toLocaleString()}</span>
      <span class="badge ctd">CTD ${ctd.toLocaleString()}</span>
      <span class="badge xbt">XBT ${xbt.toLocaleString()}</span>
    </div>
    <div class="gd-stat"><div class="lbl">Total observations</div><div class="val">${(d.n_obs || 0).toLocaleString()}</div></div>
    <div class="gd-section-title">Platform mix</div>
    <div class="pie-row"><div class="pie-ring" style="background:conic-gradient(${pie || '#333 0 100%'})"></div><div class="pie-legend">${leg}</div></div>
    ${qcPieHtml(d.qc_temp, "Raw temp_qc distribution")}
    <button class="run-btn" id="gotoQc">QC predict on this grid →</button>
    <button class="run-btn secondary" id="gotoAnalysis">Switch to Analysis →</button>`;

  document.getElementById("gotoQc").onclick = () => goQcFromGrid(bb);
  document.getElementById("gotoAnalysis").onclick = () => {
    document.querySelector('[data-mode="analysis"]').click();
    setTimeout(() => selectGrid(d.grid_id), 80);
  };
}

function renderAnalysisPanel(panel, d) {
  const bb = d.bbox;
  const th = d.thermohaline || {};
  const t = th.temp || {}, s = th.salinity || {}, dep = th.depth || {};
  const ps = d.prediction_sample || {};
  const zs = d.zscore_stats || {};

  const depthBars = (d.depth_bins || []).map((b) =>
    `<div class="bar-row"><span>${b.bin}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, b.n / (d.n_obs || 1) * 100 * 3)}%"></div></div><span>${b.n.toLocaleString()}</span></div>`
  ).join("");

  const monthBars = (d.monthly || []).map((m) =>
    `<div class="bar-row"><span>M${m.month}</span><div class="bar-track"><div class="bar-fill teal" style="width:${Math.min(100, m.n / (d.n_obs || 1) * 100 * 4)}%"></div></div><span>${m.n.toLocaleString()}</span></div>`
  ).join("");

  const predTable = (ps.rows || []).slice(0, 8).map((r) =>
    `<tr><td>${fmt(r.depth, "", 0)}</td><td>${fmt(r.temperature)}</td><td>${fmt(r.salinity)}</td>
     <td>${r.raw_temp_qc || "—"}</td><td>${r.pred_temp_label || "—"}</td>
     <td>${r.raw_psal_qc || "—"}</td><td>${r.pred_psal_label || "—"}</td></tr>`
  ).join("");

  panel.innerHTML = `
    <div class="gd-header"><span class="gid">GRID-${d.grid_id}</span><span class="mode-tag ana">Analysis</span></div>
    <div class="sub-bounds">${bb.lat_min}°–${bb.lat_max}°N · ${bb.lon_min}°–${bb.lon_max}°E</div>

    <div class="gd-section-title">Thermohaline statistics</div>
    <div class="gd-stats-grid">
      <div class="gd-stat"><div class="lbl">Temp mean</div><div class="val">${fmt(t.mean, "°C")}</div><div class="sub">${fmt(t.min)} – ${fmt(t.max, "°C")}</div></div>
      <div class="gd-stat"><div class="lbl">Sal mean</div><div class="val">${fmt(s.mean, " PSU")}</div><div class="sub">${fmt(s.min)} – ${fmt(s.max, " PSU")}</div></div>
      <div class="gd-stat"><div class="lbl">Clim T ±σ</div><div class="val">${fmt(t.clim_mean, "°C")}</div><div class="sub">σ ${fmt(t.clim_std)}</div></div>
      <div class="gd-stat"><div class="lbl">Clim S ±σ</div><div class="val">${fmt(s.clim_mean, " PSU")}</div><div class="sub">σ ${fmt(s.clim_std)}</div></div>
      <div class="gd-stat"><div class="lbl">Depth mean</div><div class="val">${fmt(dep.mean, " m", 0)}</div><div class="sub">max ${fmt(dep.max, " m", 0)}</div></div>
      <div class="gd-stat"><div class="lbl">Z-score |T|, |S|</div><div class="val">${fmt(zs.mean_abs_temp_z)} / ${fmt(zs.mean_abs_sal_z)}</div></div>
    </div>

    ${qcPieHtml(d.qc_temp, "Raw temp_qc")}
    ${qcPieHtml(d.qc_psal, "Raw psal_qc")}

    <div class="gd-section-title">Model QC vs raw (n=${ps.n || 0} sample)</div>
    <div class="agree-row">
      <span>temp_qc agreement: <b>${ps.temp_agreement_pct != null ? ps.temp_agreement_pct + "%" : "n/a"}</b></span>
      <span>psal_qc agreement: <b>${ps.psal_agreement_pct != null ? ps.psal_agreement_pct + "%" : "n/a"}</b></span>
    </div>
    <div class="model-line">${d.model?.name || "RF"} · temp ${d.model?.temp_qc_ready ? "✓" : "—"} · psal ${d.model?.psal_qc_ready ? "✓" : "—"}</div>
    ${predTable ? `<table class="results-table mini"><tr><th>z(m)</th><th>T</th><th>S</th><th>raw Tqc</th><th>pred Tqc</th><th>raw Sqc</th><th>pred Sqc</th></tr>${predTable}</table>` : ""}

    <div class="gd-section-title">Depth distribution</div>
    <div class="bar-chart">${depthBars || "<span class='sub-bounds'>No depth data</span>"}</div>

    <div class="gd-section-title">Seasonal coverage</div>
    <div class="bar-chart">${monthBars || "<span class='sub-bounds'>No month field</span>"}</div>

    <button class="run-btn" id="gotoQc">QC predict on this grid →</button>
    <button class="run-btn secondary" id="gotoObs">View observations →</button>`;

  document.getElementById("gotoQc").onclick = () => goQcFromGrid(bb);
  document.getElementById("gotoObs").onclick = () => {
    document.querySelector('[data-mode="observations"]').click();
    setTimeout(() => selectGrid(d.grid_id), 80);
  };
}

function goQcFromGrid(bb) {
  document.querySelector('[data-page="qc"]').click();
  setTimeout(() => {
    document.getElementById("qcLat").value = bb.lat_center.toFixed(3);
    document.getElementById("qcLon").value = bb.lon_center.toFixed(3);
  }, 50);
}

function fmt(v, suf = "", dec = 2) {
  if (v == null || Number.isNaN(v)) return "n/a";
  return Number(v).toFixed(dec) + suf;
}

function renderHeatmap(data) {
  const canvas = document.getElementById("heatmapCanvas");
  if (!canvas || !data.values?.length) return;
  const w = data.width, h = data.height;
  canvas.width = w * 4; canvas.height = h * 2;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(canvas.width, canvas.height);
  const mx = data.max || 1;
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const sy = Math.floor(y / canvas.height * h);
      const sx = Math.floor(x / canvas.width * w);
      const t = data.values[sy * w + sx] / mx;
      const i = (y * canvas.width + x) * 4;
      img.data[i] = 8; img.data[i + 1] = 24 + 221 * t; img.data[i + 2] = 32 + 180 * t;
      img.data[i + 3] = 40 + 200 * t;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function qcPageHtml() {
  return `
    <div class="page-head">
      <div class="eyebrow">QC Inference Station</div>
      <h2>QC Prediction Centre</h2>
      <p>Predict <b>temp_qc</b> and <b>psal_qc</b> from profile coordinates. Resolved grid is <b>highlighted on the MHOG map</b>.</p>
      <div id="modelStatus" style="font-size:12px;color:var(--ink-2);margin-top:6px"></div>
    </div>
    <div class="qc-layout">
      <div class="panel">
        <h3>Manual entry</h3>
        <div class="qc-grid">
          <div class="field"><label>Latitude</label><input id="qcLat" type="number" step="0.001" value="14.2"></div>
          <div class="field"><label>Longitude</label><input id="qcLon" type="number" step="0.001" value="72.9"></div>
          <div class="field"><label>Depth (m)</label><input id="qcDepth" type="number" value="200"></div>
          <div class="field"><label>Temperature °C</label><input id="qcTemp" type="number" step="0.01" value="19.4"></div>
          <div class="field"><label>Salinity PSU</label><input id="qcPsal" type="number" step="0.01" value="35.1"></div>
          <div class="field"><label>Month</label><input id="qcMonth" type="number" min="1" max="12" value="6"></div>
        </div>
        <div id="resolvedGrid" style="margin-top:10px;font-size:13px;color:var(--glow)"></div>
        <button class="run-btn" id="runQc">Run QC Prediction</button>
        <hr style="border-color:var(--panel-border);margin:16px 0">
        <h3>Upload batch (CSV / Parquet)</h3>
        <div class="dropzone" id="dropzone">Drop file or click to upload</div>
        <input type="file" id="fileInput" accept=".csv,.parquet" hidden>
      </div>
      <div class="panel"><h3>Results</h3><div id="qcResults"><div class="gd-empty">No predictions yet.</div></div></div>
    </div>`;
}

function wireQcPage() {
  document.getElementById("runQc").addEventListener("click", runQc);
  const dz = document.getElementById("dropzone");
  const fi = document.getElementById("fileInput");
  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => { if (fi.files[0]) uploadBatch(fi.files[0]); });
  api("/api/models").then((d) => {
    const el = document.getElementById("modelStatus");
    if (!el) return;
    el.innerHTML = d.models.map((m) =>
      `${m.target}: ${m.bundle_loaded ? "✓ ready" : m.plain_exists ? "⚠ build bundles" : "✗ not trained"}`
    ).join(" · ");
  }).catch(() => {});
}

async function runQc() {
  const body = {
    lat: +document.getElementById("qcLat").value,
    lon: +document.getElementById("qcLon").value,
    depth: +document.getElementById("qcDepth").value,
    temperature: +document.getElementById("qcTemp").value,
    salinity: +document.getElementById("qcPsal").value,
    month: +document.getElementById("qcMonth").value,
  };
  const out = document.getElementById("qcResults");
  out.innerHTML = '<div class="loading">Running pipeline…</div>';
  try {
    const res = await api("/api/qc/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    document.getElementById("resolvedGrid").textContent = `Resolved grid: GRID-${res.grid_id}`;
    out.innerHTML = qcResultHtml(res);
    goHighlightGrid(res.grid_id);
  } catch (e) {
    out.innerHTML = `<div class="gd-empty">${e.message}</div>`;
  }
}

function qcResultHtml(res) {
  const chip = (o) => {
    if (!o) return "n/a";
    const cls = (o.label || "").toLowerCase().replace(/\s+/g, "");
    const conf = o.confidence != null ? ` · ${(o.confidence * 100).toFixed(0)}%` : "";
    return `<span class="qc-chip qc-${cls}">${o.label} (${o.flag})${conf}</span>`;
  };
  return `<p><b>GRID-${res.grid_id}</b></p>
    <p class="sub-bounds">Model: ${res.model}</p>
    <p>temp_qc: ${chip(res.temp_qc)}</p>
    <p>psal_qc: ${chip(res.psal_qc)}</p>
    <p class="sub-bounds">Grid highlighted on MHOG Observatory map.</p>`;
}

async function uploadBatch(file) {
  const out = document.getElementById("qcResults");
  out.innerHTML = '<div class="loading">Processing batch…</div>';
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch(API + "/api/qc/batch", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    let rows = "<table class='results-table'><tr><th>Grid</th><th>temp_qc</th><th>psal_qc</th></tr>";
    data.results.slice(0, 30).forEach((r) => {
      rows += `<tr><td>${r.grid_id}</td><td>${r.temp_qc.label}</td><td>${r.psal_qc.label}</td></tr>`;
    });
    rows += "</table>";
    out.innerHTML = `<p>${data.count} profiles processed</p>${rows}`;
    if (data.results[0]) goHighlightGrid(data.results[0].grid_id);
  } catch (e) {
    out.innerHTML = `<div class="gd-empty">${e.message}</div>`;
  }
}

function goHighlightGrid(gid) {
  highlightGridId = gid;
  document.querySelector('[data-page="grid"]').click();
  setTimeout(() => {
    selectGrid(gid);
    document.querySelectorAll(".gcell").forEach((el) => {
      if (parseInt(el.dataset.id, 10) === gid) el.classList.add("highlighted");
    });
  }, 250);
}

window.highlightGrid = goHighlightGrid;
