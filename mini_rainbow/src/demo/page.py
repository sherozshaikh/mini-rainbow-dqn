"""Self-contained HTML page for the live demo dashboard."""

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini-Rainbow DQN &mdash; Live Demo</title>
<style>
:root {
    --bg: #0d1117; --bg2: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --dim: #484f58;
    --blue: #58a6ff; --green: #3fb950; --orange: #d29922; --red: #f85149;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif; }

/* Header */
.hdr { background:var(--bg2); border-bottom:1px solid var(--border); padding:10px 24px; display:flex; align-items:center; gap:12px; }
.hdr h1 { font-size:17px; color:var(--blue); font-weight:600; }
.hdr .badge { background:#238636; color:#fff; font-size:10px; padding:2px 8px; border-radius:10px; font-weight:600; }
.hdr .sub { font-size:12px; color:var(--muted); margin-left:auto; }

/* Main layout: two game columns + bottom metrics */
.main { display:grid; grid-template-columns:1fr 1fr; gap:0; height:calc(100vh - 45px); }

/* Each agent column */
.agent-col { border-right:1px solid var(--border); display:flex; flex-direction:column; overflow-y:auto; }
.agent-col:last-child { border-right:none; }

/* Agent header */
.agent-hdr { background:var(--bg2); border-bottom:1px solid var(--border); padding:8px 16px; display:flex; align-items:center; gap:8px; }
.agent-hdr .name { font-size:14px; font-weight:600; }
.agent-hdr .arch { font-size:11px; color:var(--muted); background:var(--bg); padding:2px 6px; border-radius:3px; border:1px solid var(--border); }

/* Game area */
.game-area { display:flex; align-items:center; justify-content:center; padding:12px; background:#000; }
.game-area img { border-radius:4px; image-rendering:pixelated; }

/* Score banner */
.score-row { display:flex; justify-content:center; align-items:baseline; gap:8px; padding:6px 0; background:var(--bg2); border-bottom:1px solid var(--border); }
.score-big { font-size:32px; font-weight:700; }
.score-label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }

/* Stat cards */
.cards { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; padding:10px 12px; }
.card { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:10px 12px; }
.card .lbl { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }
.card .val { font-size:20px; font-weight:700; margin-top:2px; }
.card .val.b { color:var(--blue); }
.card .val.g { color:var(--green); }
.card .val.o { color:var(--orange); }

/* Q-values */
.q-section { padding:8px 12px; }
.q-title { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; border-bottom:1px solid var(--border); padding-bottom:4px; }
.q-row { display:flex; align-items:center; gap:6px; margin:4px 0; }
.q-lbl { width:44px; font-size:11px; color:var(--muted); text-align:right; }
.q-track { flex:1; height:16px; background:#21262d; border-radius:3px; overflow:hidden; }
.q-fill { height:100%; border-radius:3px; transition:width .12s ease; }
.q-fill.best { background:linear-gradient(90deg,#1f6feb,#58a6ff); }
.q-fill.other { background:#30363d; }
.q-num { width:60px; font-size:11px; color:var(--text); font-family:'Courier New',monospace; text-align:right; }

/* Mini chart */
.chart-wrap { padding:8px 12px; }
.chart-wrap canvas { width:100%; height:80px; display:block; background:var(--bg2); border:1px solid var(--border); border-radius:4px; }

/* Footer */
.ftr { padding:8px 12px; font-size:10px; color:var(--dim); display:flex; gap:12px; flex-wrap:wrap; }
.ftr a { color:var(--blue); text-decoration:none; }
</style>
</head>
<body>

<div class="hdr">
    <h1>Mini-Rainbow DQN</h1>
    <span class="badge">LIVE</span>
    <span class="sub">Side-by-side comparison &mdash; ALE/Breakout-v5</span>
</div>

<div class="main">
    <!-- DQN column -->
    <div class="agent-col" id="col-DQN">
        <div class="agent-hdr">
            <span class="name" style="color:var(--blue);">DQN</span>
            <span class="arch">Standard Q-Network &bull; 5M steps</span>
        </div>
        <div class="game-area"><img id="frame-DQN" width="240" height="320" alt="DQN"></div>
        <div class="score-row">
            <span class="score-big" style="color:var(--blue);" id="score-DQN">0</span>
            <span class="score-label">Score</span>
        </div>
        <div class="cards">
            <div class="card"><div class="lbl">Episode</div><div class="val b" id="ep-DQN">0</div></div>
            <div class="card"><div class="lbl">Avg (10 ep)</div><div class="val g" id="avg-DQN">0</div></div>
            <div class="card"><div class="lbl">Best</div><div class="val o" id="best-DQN">0</div></div>
        </div>
        <div class="q-section">
            <div class="q-title">Q-Values</div>
            <div class="q-row"><span class="q-lbl">NOOP</span><div class="q-track"><div class="q-fill other" id="q-DQN-0"></div></div><span class="q-num" id="qv-DQN-0">0</span></div>
            <div class="q-row"><span class="q-lbl">FIRE</span><div class="q-track"><div class="q-fill other" id="q-DQN-1"></div></div><span class="q-num" id="qv-DQN-1">0</span></div>
            <div class="q-row"><span class="q-lbl">RIGHT</span><div class="q-track"><div class="q-fill other" id="q-DQN-2"></div></div><span class="q-num" id="qv-DQN-2">0</span></div>
            <div class="q-row"><span class="q-lbl">LEFT</span><div class="q-track"><div class="q-fill other" id="q-DQN-3"></div></div><span class="q-num" id="qv-DQN-3">0</span></div>
        </div>
        <div class="chart-wrap"><canvas id="chart-DQN" height="80"></canvas></div>
        <div class="ftr">
            <span>CNN: 3-conv + 2-fc</span>
            <span>Replay: Uniform</span>
            <span>Target: argmax(Q_target)</span>
        </div>
    </div>

    <!-- Rainbow-Lite column -->
    <div class="agent-col" id="col-Rainbow-Lite">
        <div class="agent-hdr">
            <span class="name" style="color:var(--green);">Rainbow-Lite</span>
            <span class="arch">Dueling DDQN + PER &bull; 2M steps</span>
        </div>
        <div class="game-area"><img id="frame-Rainbow-Lite" width="240" height="320" alt="Rainbow-Lite"></div>
        <div class="score-row">
            <span class="score-big" style="color:var(--green);" id="score-Rainbow-Lite">0</span>
            <span class="score-label">Score</span>
        </div>
        <div class="cards">
            <div class="card"><div class="lbl">Episode</div><div class="val b" id="ep-Rainbow-Lite">0</div></div>
            <div class="card"><div class="lbl">Avg (10 ep)</div><div class="val g" id="avg-Rainbow-Lite">0</div></div>
            <div class="card"><div class="lbl">Best</div><div class="val o" id="best-Rainbow-Lite">0</div></div>
        </div>
        <div class="q-section">
            <div class="q-title">Q-Values</div>
            <div class="q-row"><span class="q-lbl">NOOP</span><div class="q-track"><div class="q-fill other" id="q-Rainbow-Lite-0"></div></div><span class="q-num" id="qv-Rainbow-Lite-0">0</span></div>
            <div class="q-row"><span class="q-lbl">FIRE</span><div class="q-track"><div class="q-fill other" id="q-Rainbow-Lite-1"></div></div><span class="q-num" id="qv-Rainbow-Lite-1">0</span></div>
            <div class="q-row"><span class="q-lbl">RIGHT</span><div class="q-track"><div class="q-fill other" id="q-Rainbow-Lite-2"></div></div><span class="q-num" id="qv-Rainbow-Lite-2">0</span></div>
            <div class="q-row"><span class="q-lbl">LEFT</span><div class="q-track"><div class="q-fill other" id="q-Rainbow-Lite-3"></div></div><span class="q-num" id="qv-Rainbow-Lite-3">0</span></div>
        </div>
        <div class="chart-wrap"><canvas id="chart-Rainbow-Lite" height="80"></canvas></div>
        <div class="ftr">
            <span>Dueling CNN: V(s) + A(s,a)</span>
            <span>Replay: PER (alpha=0.6)</span>
            <span>Target: argmax(Q_online)</span>
        </div>
    </div>
</div>

<script>
const agents = ['DQN', 'Rainbow-Lite'];
const colors = {'DQN': '#58a6ff', 'Rainbow-Lite': '#3fb950'};
const histories = {'DQN': [], 'Rainbow-Lite': []};
const lastEp = {'DQN': 0, 'Rainbow-Lite': 0};
const MAX_H = 50;

function drawChart(name) {
    const c = document.getElementById('chart-' + name);
    if (!c) return;
    const ctx = c.getContext('2d');
    const w = c.width = c.offsetWidth;
    const h = c.height = 80;
    ctx.clearRect(0, 0, w, h);
    const pts = histories[name];
    if (pts.length < 2) return;
    const mx = Math.max(...pts, 1);
    const step = w / (MAX_H - 1);
    const start = Math.max(0, pts.length - MAX_H);

    // Grid
    ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
    for (let y = 20; y < h; y += 20) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }

    // Line
    ctx.beginPath(); ctx.strokeStyle = colors[name]; ctx.lineWidth = 2;
    for (let i = start; i < pts.length; i++) {
        const x = (i - start) * step;
        const y = h - (pts[i] / mx) * (h - 8) - 4;
        i === start ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill
    ctx.lineTo((pts.length - 1 - start) * step, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const col = colors[name];
    ctx.fillStyle = col.replace(')', ',0.07)').replace('rgb', 'rgba').replace('#', '');
    // Simpler: just use a semi-transparent version
    ctx.fillStyle = name === 'DQN' ? 'rgba(88,166,255,0.07)' : 'rgba(63,185,80,0.07)';
    ctx.fill();

    ctx.fillStyle = '#8b949e'; ctx.font = '9px monospace';
    ctx.fillText(mx.toFixed(0), 3, 10);
}

function updateAgent(name, d) {
    const f = document.getElementById('frame-' + name);
    if (f && d.frame) f.src = 'data:image/jpeg;base64,' + d.frame;

    const s = document.getElementById('score-' + name);
    if (s) s.textContent = d.score;

    const ep = document.getElementById('ep-' + name);
    if (ep) ep.textContent = d.episode;

    const avg = document.getElementById('avg-' + name);
    if (avg) avg.textContent = d.avg_score;

    const best = document.getElementById('best-' + name);
    if (best) best.textContent = d.best_score;

    // Q-values
    if (d.q_values && d.q_values.length === 4) {
        const qv = d.q_values;
        const mx = Math.max(...qv.map(Math.abs), 0.01);
        const bi = qv.indexOf(Math.max(...qv));
        for (let i = 0; i < 4; i++) {
            const pct = Math.max((qv[i] / mx) * 50 + 50, 3);
            const bar = document.getElementById('q-' + name + '-' + i);
            if (bar) { bar.style.width = pct + '%'; bar.className = 'q-fill ' + (i === bi ? 'best' : 'other'); }
            const vl = document.getElementById('qv-' + name + '-' + i);
            if (vl) vl.textContent = qv[i].toFixed(3);
        }
    }

    // Chart
    if (d.episode > lastEp[name] && d.recent_scores && d.recent_scores.length > 0) {
        lastEp[name] = d.episode;
        histories[name].push(d.recent_scores[d.recent_scores.length - 1]);
        drawChart(name);
    }
}

const src = new EventSource('/stream');
src.onmessage = function(e) {
    const data = JSON.parse(e.data);
    for (const name of agents) {
        if (data[name]) updateAgent(name, data[name]);
    }
};

window.addEventListener('resize', () => agents.forEach(drawChart));
</script>
</body>
</html>"""
