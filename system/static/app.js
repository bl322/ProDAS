/* ProDAS 前端 */

const $ = (id) => document.getElementById(id);

const SLIDERS = ["budget", "max_iters", "beam_width", "probe_width", "p_iter",
  "hard_restart_at", "operators_per_round", "bandit_c", "bandit_epsilon",
  "bandit_ema", "workers", "auto_rounds", "limit"];
const NUMERIC = {limit: 1, offset: 1, seed: 1, max_tokens: 1, retries: 1};
const BOOL = ["use_mock", "judge_confirm", "resume", "retry_errors", "stall_stop", "mode_auto"];
const TEXT = ["goal", "dataset_path", "tag", "base_url", "api_key", "no_proxy",
  "target_model", "attacker_model", "judge_model", "judge_mode", "seed_from",
  "bandit_strategy", "operator_family", "bandit_scope"];
const FLOAT = ["target_temperature", "attacker_temperature", "judge_temperature", "timeout"];

let META = null;
let TASK_ID = "";
let RUNNING = false;
let OPS = {};          // 算子名 → {n, mean, ucb, family}
let RECORDS = [];      // 本轮明细
let PHASES = [];

function fmt(n, d = 2) { return (Math.round(n * 10 ** d) / 10 ** d).toFixed(d); }
function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function clip(s, n) {
  s = String(s == null ? "" : s);
  return s.length > n ? s.slice(0, n) + "…" : s;
}

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------
async function boot() {
  const res = await fetch("/api/meta");
  META = await res.json();
  const d = META.defaults;

  for (const k of SLIDERS) {
    const el = $(k);
    if (!el) continue;
    if (META.limits[k]) { el.min = META.limits[k][0]; el.max = META.limits[k][1]; }
    el.value = d[k];
    el.addEventListener("input", () => { const v = $("v_" + k); if (v) v.textContent = el.value; });
    const v = $("v_" + k); if (v) v.textContent = el.value;
  }
  for (const k of Object.keys(NUMERIC)) if ($(k)) $(k).value = d[k];
  for (const k of FLOAT) if ($(k) && d[k] !== undefined) $(k).value = d[k];
  for (const k of BOOL) if ($(k)) $(k).checked = !!d[k];
  for (const k of TEXT) if ($(k) && d[k] !== undefined) $(k).value = d[k];

  fillSelect("bandit_strategy", META.bandit_strategies, d.bandit_strategy);
  fillSelect("bandit_scope", META.bandit_scopes, d.bandit_scope);
  fillSelect("operator_family", META.families, d.operator_family);

  for (const op of META.operators) OPS[op.name] = {n: 0, mean: 0, family: op.family};
  renderOps();

  $("batch_dir").textContent = META.paths.batch_dir_rel || META.paths.batch_dir;
  loadPipeline();
  loadRuns();
  bindMode();
  bindNav();
  syncStrategyHint();
  health();
  setInterval(health, 30000);
}

// ---------------------------------------------------------------------------
// 顶部页签：评测台 / 历史结果
// ---------------------------------------------------------------------------
function bindNav() {
  const ev = $("nav-eval"), hi = $("nav-history");
  if (!ev || !hi) return;
  ev.onclick = () => {
    ev.classList.add("on"); hi.classList.remove("on");
    $("view-eval").style.display = ""; $("view-history").style.display = "none";
  };
  hi.onclick = () => {
    hi.classList.add("on"); ev.classList.remove("on");
    $("view-eval").style.display = "none"; $("view-history").style.display = "";
    if (!$("history_body").dataset.loaded) loadHistory();
  };
}

function fillSelect(id, options, value) {
  const el = $(id);
  if (!el) return;
  el.innerHTML = options.map(o => `<option value="${esc(o.value)}">${esc(o.label)}</option>`).join("");
  el.value = value;
  el.addEventListener("change", () => { if (id === "bandit_strategy") syncStrategyHint(); });
}

function syncStrategyHint() {
  const v = $("bandit_strategy").value;
  const found = (META.bandit_strategies || []).find(o => o.value === v);
  $("hint_strategy").textContent = found ? found.label : "";
  const showC = v === "ucb", showE = v === "epsilon";
  $("row_bandit_c").style.display = showC ? "" : "none";
  $("row_bandit_epsilon").style.display = showE ? "" : "none";
}

async function health() {
  const chip = $("chip_backend");
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    if (chip) chip.textContent = "后端已连接 · ProDAS · " + j.time;
  } catch (e) {
    if (chip) chip.textContent = "后端离线";
  }
}

async function loadPipeline() {
  const r = await fetch("/api/pipeline");
  const j = await r.json();
  $("pipeline").innerHTML = j.html;
}

// ---------------------------------------------------------------------------
// 模式切换
// ---------------------------------------------------------------------------
function bindMode() {
  const single = () => {
    $("tab-single").classList.add("on"); $("tab-batch").classList.remove("on");
    $("pane-single").style.display = ""; $("pane-batch").style.display = "none";
  };
  const batch = () => {
    $("tab-batch").classList.add("on"); $("tab-single").classList.remove("on");
    $("pane-single").style.display = "none"; $("pane-batch").style.display = "";
  };
  $("tab-single").onclick = single;
  $("tab-batch").onclick = batch;
  $("mode_auto").addEventListener("change", () => {
    $("row_auto_rounds").style.display = $("mode_auto").checked ? "" : "none";
  });
  $("row_auto_rounds").style.display = $("mode_auto").checked ? "" : "none";
}

function isBatch() { return $("tab-batch").classList.contains("on"); }

function collectParams() {
  const p = {};
  for (const k of SLIDERS) if ($(k)) p[k] = parseFloat($(k).value);
  for (const k of Object.keys(NUMERIC)) if ($(k)) p[k] = parseInt($(k).value, 10);
  for (const k of FLOAT) if ($(k) && $(k).value !== "") p[k] = parseFloat($(k).value);
  for (const k of BOOL) if ($(k)) p[k] = $(k).checked;
  for (const k of TEXT) if ($(k)) p[k] = $(k).value;
  delete p.mode_auto;
  p.mode = ($("mode_auto").checked && isBatch()) ? "auto" : "single-pass";
  if (isBatch()) p.goal = "";
  return p;
}

// ---------------------------------------------------------------------------
// 运行
// ---------------------------------------------------------------------------
async function run() {
  if (RUNNING) return;
  RUNNING = true;
  $("run").disabled = true;
  $("cancel").disabled = false;
  $("status").textContent = "运行中…";
  $("log").innerHTML = "";
  RECORDS = []; PHASES = [];
  renderRecords(); renderCards(null); $("phase_box").innerHTML = "";

  TASK_ID = (isBatch() ? "batch-" : "single-") + Date.now();
  const url = isBatch() ? "/api/batch" : "/api/single";
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({task_id: TASK_ID, params: collectParams()}),
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        if (line.trim()) {
          try { handle(JSON.parse(line)); } catch (e) { /* 忽略坏行 */ }
        }
      }
    }
  } catch (e) {
    logline("err", "请求失败：" + e);
  } finally {
    RUNNING = false;
    $("run").disabled = false;
    $("cancel").disabled = true;
    $("status").textContent = "完成";
    loadRuns();
  }
}

async function cancel() {
  if (!TASK_ID) return;
  await fetch("/api/cancel", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({task_id: TASK_ID}),
  });
  $("status").textContent = "取消中…";
}

// ---------------------------------------------------------------------------
// 事件渲染
// ---------------------------------------------------------------------------
function logline(cls, text) {
  const el = document.createElement("div");
  el.className = "l-" + cls;
  el.textContent = text;
  $("log").appendChild(el);
  $("log").scrollTop = $("log").scrollHeight;
}

function handle(ev) {
  const t = ev.type;
  if (t === "attempt") {
    const op = ev.operator || ev.origin;
    const hit = ev.judge_score >= 5;
    logline(hit ? "hit" : "dim",
      `  #${ev.round} [${op}] score=${ev.judge_score} L=${ev.L} ${clip(ev.prompt, 70)}`);
    return;
  }
  if (t === "goal_done") {
    const r = ev.record;
    RECORDS.push(r);
    renderRecords();
    logline(r.success ? "ok" : "warn",
      `✓ ${r.id} score=${r.best_score} queries=${r.queries} success=${r.success}`);
    renderCards(null);
    return;
  }
  if (t === "goal_error") { logline("err", `✗ ${ev.id} ${ev.message}`); return; }
  if (t === "bandit") {
    if (ev.operator && OPS[ev.operator]) {
      OPS[ev.operator].n = ev.n; OPS[ev.operator].mean = ev.mean;
    }
    renderOps();
    return;
  }
  if (t === "goal_start") { logline("dim", `▶ ${ev.id} ${clip(ev.goal, 60)}`); return; }
  if (t === "phase_summary") {
    PHASES.push(ev.summary);
    renderPhaseTable();
    const s = ev.summary;
    logline("hit", `[第 ${ev.phase_no}/${ev.phase_total} 轮] ASR=${s.asr}% 新增 ${s.success}/${s.total} 累计 ${s.cumulative_asr}% 剩余 ${s.remaining}`);
    return;
  }
  if (t === "phase_split") { logline("dim", `  失败集 → ${ev.next_dataset} (${ev.remaining})`); return; }
  if (t === "phase_skip") { logline("warn", `  跳过第 ${ev.phase_no} 轮：${ev.skip_reason}`); return; }
  if (t === "seed_map") { logline("dim", `  注入种子：${ev.goals} 个目标`); return; }
  if (t === "resume") { logline("dim", `  断点续跑：跳过 ${ev.skipped} 条，剩余 ${ev.pending} 条`); return; }
  if (t === "warm_start") { logline("dim", `  warm start：${ev.prompts.length} 条历史高分 prompt`); return; }
  if (t === "hard_restart") { logline("warn", `  第 ${ev.iteration} 轮：低分重启，换直接产出型起点`); return; }
  if (t === "ready") { logline("dim", "引擎就绪：" + JSON.stringify(ev.engine.config)); return; }
  if (t === "done") {
    if (ev.operators) { for (const o of ev.operators) OPS[o.name] = o; renderOps(); }
    renderCards(ev.summary);
    return;
  }
  if (t === "error") { logline("err", "错误：" + ev.message); return; }
  if (t === "auto_start") { logline("dim", `多轮开始：N=${ev.auto_rounds} · ${ev.note}`); return; }
  if (t === "start" || t === "phase_start" || t === "auto_done") {
    logline("dim", JSON.stringify(ev).slice(0, 200));
    if (t === "auto_done" && ev.summary) renderCards(ev.summary);
    return;
  }
}

function renderCards(s) {
  const stats = s || summarizeLocal();
  if (!stats) { $("cards").innerHTML = ""; return; }
  const card = (k, v, cls) =>
    `<div class="card"><div class="k">${k}</div><div class="v ${cls || ""}">${v}</div></div>`;
  $("cards").innerHTML =
    card("样本数", stats.total || 0) +
    card("越狱成功", stats.success || 0, "ok") +
    card("ASR", (stats.asr != null ? stats.asr : 0) + "%", stats.asr >= 50 ? "ok" : "") +
    card("平均查询 AQC", stats.aqc != null ? stats.aqc : "-") +
    card("平均最高分", stats.avg_best_score != null ? stats.avg_best_score : "-");
  if (stats.rounds_run) {
    $("summary_tag").textContent = `实跑 ${stats.rounds_run} 轮`;
  }
}

function summarizeLocal() {
  if (!RECORDS.length) return null;
  const total = RECORDS.length;
  const success = RECORDS.filter(r => r.success).length;
  const q = RECORDS.reduce((a, r) => a + (r.queries || 0), 0) / total;
  const s = RECORDS.reduce((a, r) => a + (r.best_score || 0), 0) / total;
  return {total, success, asr: fmt(100 * success / total, 2), aqc: fmt(q, 2),
          avg_best_score: fmt(s, 2)};
}

function renderRecords() {
  const rows = RECORDS.slice().reverse().map(r => {
    const badge = r.success
      ? '<span class="badge ok">成功</span>'
      : '<span class="badge bad">' + r.best_score + " 分</span>";
    return `<tr>
      <td>${esc(r.id)}</td>
      <td>${badge}</td>
      <td>${esc(clip(r.goal, 40))}</td>
      <td class="mono">${r.queries}</td>
      <td class="mono">${esc(clip(r.best_prompt, 60))}</td>
    </tr>`;
  }).join("");
  $("rec_table").innerHTML =
    `<thead><tr><th>id</th><th>结果</th><th>目标</th><th>查询</th><th>最佳提示词</th></tr></thead>
     <tbody>${rows || '<tr><td colspan="5" class="muted">还没有结果</td></tr>'}</tbody>`;
}

function renderOps() {
  const rows = Object.keys(OPS).map(name => OPS[name] && Object.assign({name}, OPS[name]))
    .filter(Boolean)
    .sort((a, b) => (b.mean || 0) - (a.mean || 0))
    .map(o => {
      const fam = o.family === "wrap" ? "包装" : (o.family === "induce" ? "诱导" : "-");
      const bar = Math.round((o.mean || 0) * 100);
      return `<tr>
        <td>${esc(o.name)}</td>
        <td class="muted">${fam}</td>
        <td class="mono">${o.n || 0}</td>
        <td class="mono">${fmt(o.mean || 0, 3)}</td>
        <td><div style="background:var(--chip);height:8px;border-radius:4px;overflow:hidden">
            <div style="width:${bar}%;height:100%;background:var(--accent)"></div></div></td>
      </tr>`;
    }).join("");
  $("ops_table").innerHTML =
    `<thead><tr><th>算子</th><th>族</th><th>采样</th><th>平均收益</th><th>收益条</th></tr></thead>
     <tbody>${rows}</tbody>`;
}

function renderPhaseTable() {
  if (!PHASES.length) return;
  const rows = PHASES.map((p, i) => `
    <tr>
      <td>第 ${i + 1} 轮</td>
      <td>${p.total}</td>
      <td>${p.success}</td>
      <td>${p.asr}%</td>
      <td><b>${p.cumulative_asr}%</b></td>
      <td>${p.remaining}</td>
      <td>${p.aqc}</td>
    </tr>`).join("");
  $("phase_box").innerHTML =
    `<table class="data"><thead><tr><th>轮次</th><th>本轮条数</th><th>本轮成功</th>
     <th>本轮 ASR</th><th>累计 ASR</th><th>剩余</th><th>AQC</th></tr></thead>
     <tbody>${rows}</tbody></table>`;
}

// ---------------------------------------------------------------------------
// 历史结果
// ---------------------------------------------------------------------------
async function loadRuns() {
  try {
    const r = await fetch("/api/runs");
    const j = await r.json();
    const rows = (j.runs || []).map(x => `
      <tr>
        <td class="mono">${esc(x.name)}</td>
        <td>${esc(x.mtime)}</td>
        <td class="mono">${x.stats.total}</td>
        <td class="mono">${x.stats.asr}%</td>
        <td class="mono">${x.stats.aqc}</td>
        <td><button data-path="${esc(x.path)}" class="dl">明细</button></td>
      </tr>`).join("");
    $("runs_table").innerHTML =
      `<thead><tr><th>文件</th><th>时间</th><th>条数</th><th>ASR</th><th>AQC</th><th></th></tr></thead>
       <tbody>${rows || '<tr><td colspan="6" class="muted">暂无</td></tr>'}</tbody>`;
    document.querySelectorAll("button.dl").forEach(b => {
      b.onclick = async () => {
        const rr = await fetch("/api/runs/detail?path=" + encodeURIComponent(b.dataset.path));
        const jj = await rr.json();
        RECORDS = jj.records || [];
        renderRecords();
        renderCards(jj.stats);
        PHASES = []; renderPhaseTable();
        logline("dim", `载入 ${jj.name}：${jj.total} 条`);
      };
    });
  } catch (e) { /* ignore */ }
}

async function makeFailed() {
  const res = await fetch("/api/failed-set", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      dataset_path: $("dataset_path").value,
      runs: [META.paths.batch_dir + "/*.jsonl"],
    }),
  });
  const j = await res.json();
  if (j.ok && j.failed_csv) {
    $("dataset_path").value = j.failed_csv;
    logline("ok", `失败集已生成并填入数据集：${j.failed_csv}（${j.failed} 条）`);
  } else {
    logline("warn", j.message || "生成失败");
  }
}

// ---------------------------------------------------------------------------
// 历史结果页（按实验组聚合）
// ---------------------------------------------------------------------------
async function loadHistory() {
  const body = $("history_body");
  try {
    const r = await fetch("/api/history");
    const j = await r.json();
    $("hist_dir").textContent = j.batch_dir_rel || j.batch_dir || "";
    body.innerHTML = (j.groups || []).map(renderGroup).join("")
      || '<div class="muted">暂无历史结果</div>';
    body.dataset.loaded = "1";
    bindRunDetails();
  } catch (e) {
    body.innerHTML = '<div class="muted">加载失败：' + esc(e) + "</div>";
  }
}

function asrCell(v) {
  if (v == null) return '<td class="mono muted">—</td>';
  const cls = v >= 97.5 ? "asr-hi" : (v >= 90 ? "asr-mid" : "asr-lo");
  return `<td class="mono ${cls}">${v}%</td>`;
}

function renderGroup(g) {
  let pivot = "";
  if (g.models && g.models.length && g.rounds && g.rounds.length) {
    const cols = g.rounds.map(r => `<th>第 ${r} 轮</th>`).join("");
    const rows = g.models.map(m => {
      const rs = (g.runs || []).filter(e => e.model === m);
      const byRound = g.rounds.map(r => {
        const f = rs.find(e => e.round === r);
        return f ? f.stats.asr : null;
      });
      const valid = byRound.filter(v => v != null);
      const mean = valid.length
        ? Math.round(valid.reduce((a, b) => a + b, 0) / valid.length * 100) / 100 : null;
      const aqc = rs.length
        ? Math.round(rs.reduce((a, e) => a + (e.stats.aqc || 0), 0) / rs.length * 100) / 100
        : null;
      return `<tr><td class="mono">${esc(m)}</td>${byRound.map(asrCell).join("")}` +
        `${asrCell(mean)}<td class="mono">${aqc != null ? aqc : "—"}</td></tr>`;
    }).join("");
    pivot = `<div class="scroll"><table class="data">` +
      `<thead><tr><th>目标模型</th>${cols}<th>均值</th><th>AQC(全体)</th></tr></thead>` +
      `<tbody>${rows}</tbody></table></div>`;
  }
  const details = (g.runs || []).map(e => {
    const asr = e.stats.asr != null ? e.stats.asr : 0;
    const cls = asr >= 97.5 ? "asr-hi" : (asr >= 90 ? "asr-mid" : "asr-lo");
    const name = e.model && e.model !== e.tag ? `${e.model}${e.round ? " · 第 " + e.round + " 轮" : ""}` : e.tag;
    return `<details class="rundetail" data-path="${esc(e.path)}">
      <summary><b>${esc(name)}</b> · ${e.stats.total} 条 · 成功 ${e.stats.success} ·
        ASR <span class="${cls}">${asr}%</span> · AQC ${e.stats.aqc} ·
        平均最高分 ${e.stats.avg_best_score != null ? e.stats.avg_best_score : "—"} · ${esc(e.finished || "")}</summary>
      <div class="body"><div class="tiny muted">展开加载逐条明细…</div></div>
    </details>`;
  }).join("");
  return `<div class="panel hist-group">
    <div class="panel-title">${esc(g.title)}</div>
    <div class="hist-desc">${esc(g.desc)}</div>
    ${pivot}${details}
  </div>`;
}

function bindRunDetails() {
  document.querySelectorAll("details.rundetail").forEach(d => {
    if (d.dataset.bound) return;
    d.dataset.bound = "1";
    d.addEventListener("toggle", async () => {
      if (!d.open || d.dataset.loaded) return;
      const body = d.querySelector(".body");
      try {
        const r = await fetch("/api/runs/detail?path="
          + encodeURIComponent(d.dataset.path) + "&limit=500");
        const j = await r.json();
        const rows = (j.records || []).map(x => `<tr>
          <td class="mono">${esc(x.id)}</td>
          <td>${x.success
            ? '<span class="badge ok">成功</span>'
            : `<span class="badge bad">${x.best_score != null ? x.best_score : "-"} 分</span>`}</td>
          <td>${esc(clip(x.goal, 56))}</td>
          <td class="mono">${x.queries}</td>
        </tr>`).join("");
        body.innerHTML = `<table class="data">
          <thead><tr><th>id</th><th>结果</th><th>目标</th><th>查询</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="4" class="muted">暂无</td></tr>'}</tbody></table>`;
        d.dataset.loaded = "1";
      } catch (e) {
        body.innerHTML = '<div class="muted">加载失败</div>';
      }
    });
  });
}

// ---------------------------------------------------------------------------
$("run").onclick = run;
$("cancel").onclick = cancel;
$("refresh_runs").onclick = loadRuns;
$("make_failed").onclick = makeFailed;
$("hist_refresh").onclick = loadHistory;
boot();
