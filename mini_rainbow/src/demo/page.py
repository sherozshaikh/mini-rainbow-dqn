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
.hdr .sub { font-size:12px; color:var(--muted); }
.hdr .spacer { flex:1; }

/* Speed controls */
.speed-bar { display:flex; align-items:center; gap:6px; }
.speed-bar span { font-size:11px; color:var(--muted); }
.speed-btn { background:var(--bg); border:1px solid var(--border); color:var(--muted); padding:3px 10px; border-radius:4px; font-size:12px; cursor:pointer; font-family:inherit; }
.speed-btn:hover { border-color:var(--blue); color:var(--text); }
.speed-btn.active { background:var(--blue); color:#fff; border-color:var(--blue); }

/* Main layout */
.main { display:grid; grid-template-columns:1fr 1fr; gap:0; height:calc(100vh - 45px - 180px); overflow-y:auto; }

/* Agent column */
.agent-col { border-right:1px solid var(--border); display:flex; flex-direction:column; overflow-y:auto; }
.agent-col:last-child { border-right:none; }

.agent-hdr { background:var(--bg2); border-bottom:1px solid var(--border); padding:8px 16px; display:flex; align-items:center; gap:8px; }
.agent-hdr .name { font-size:14px; font-weight:600; }
.agent-hdr .arch { font-size:11px; color:var(--muted); background:var(--bg); padding:2px 6px; border-radius:3px; border:1px solid var(--border); }

.game-area { display:flex; align-items:center; justify-content:center; padding:12px; background:#000; }
.game-area img { border-radius:4px; image-rendering:pixelated; }

.score-row { display:flex; justify-content:center; align-items:baseline; gap:8px; padding:6px 0; background:var(--bg2); border-bottom:1px solid var(--border); }
.score-big { font-size:32px; font-weight:700; }
.score-label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }

/* Stat cards */
.cards { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; padding:10px 12px; }
.card { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:10px 12px; }
.card .lbl { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }
.card .val { font-size:20px; font-weight:700; margin-top:2px; }
.card .val.b { color:var(--blue); }
.card .val.g { color:var(--green); }
.card .val.o { color:var(--orange); }

/* Q-values */
.section { padding:8px 12px; }
.section-title { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; border-bottom:1px solid var(--border); padding-bottom:4px; }
.q-row { display:flex; align-items:center; gap:6px; margin:4px 0; }
.q-lbl { width:44px; font-size:11px; color:var(--muted); text-align:right; }
.q-track { flex:1; height:16px; background:#21262d; border-radius:3px; overflow:hidden; }
.q-fill { height:100%; border-radius:3px; transition:width .12s ease; }
.q-fill.best { background:linear-gradient(90deg,#1f6feb,#58a6ff); }
.q-fill.other { background:#30363d; }
.q-num { width:60px; font-size:11px; color:var(--text); font-family:'Courier New',monospace; text-align:right; }

/* Action distribution */
.action-row { display:flex; align-items:center; gap:6px; margin:3px 0; }
.action-lbl { width:44px; font-size:11px; color:var(--muted); text-align:right; }
.action-track { flex:1; height:14px; background:#21262d; border-radius:3px; overflow:hidden; }
.action-fill { height:100%; border-radius:3px; transition:width .3s ease; }
.action-pct { width:40px; font-size:11px; color:var(--muted); font-family:'Courier New',monospace; text-align:right; }

/* Combined episode log */
.combined-log { background:var(--bg2); border-top:1px solid var(--border); }
.combined-log-wrap { max-height:160px; overflow-y:auto; padding:0 16px 8px; }
.ep-table { width:100%; border-collapse:collapse; font-size:11px; font-family:'Courier New',monospace; }
.ep-table th { text-align:left; color:var(--muted); font-weight:normal; text-transform:uppercase; font-size:10px; letter-spacing:0.5px; padding:4px 8px; border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--bg2); }
.ep-table td { padding:3px 8px; border-bottom:1px solid #21262d; }
.ep-log-empty { font-size:11px; color:var(--dim); padding:8px 0; }
.ep-row-dqn td { color:var(--blue); }
.ep-row-rl td { color:var(--green); }

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
    <span class="spacer"></span>
    <div class="speed-bar">
        <span>Speed:</span>
        <button class="speed-btn active" onclick="setSpeed(1)">1x</button>
        <button class="speed-btn" onclick="setSpeed(10)">10x</button>
        <button class="speed-btn" onclick="setSpeed(30)">30x</button>
    </div>
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
            <div class="card"><div class="lbl">Lives / Lost</div><div class="val" id="lives-DQN" style="font-size:16px;">5 / 0</div></div>
        </div>
        <div class="section">
            <div class="section-title">Q-Values</div>
            <div class="q-row"><span class="q-lbl">NOOP</span><div class="q-track"><div class="q-fill other" id="q-DQN-0"></div></div><span class="q-num" id="qv-DQN-0">0</span></div>
            <div class="q-row"><span class="q-lbl">FIRE</span><div class="q-track"><div class="q-fill other" id="q-DQN-1"></div></div><span class="q-num" id="qv-DQN-1">0</span></div>
            <div class="q-row"><span class="q-lbl">RIGHT</span><div class="q-track"><div class="q-fill other" id="q-DQN-2"></div></div><span class="q-num" id="qv-DQN-2">0</span></div>
            <div class="q-row"><span class="q-lbl">LEFT</span><div class="q-track"><div class="q-fill other" id="q-DQN-3"></div></div><span class="q-num" id="qv-DQN-3">0</span></div>
        </div>
        <div class="section">
            <div class="section-title">Action Distribution (this episode)</div>
            <div class="action-row"><span class="action-lbl">NOOP</span><div class="action-track"><div class="action-fill" id="ad-DQN-0" style="width:0%;background:var(--dim);"></div></div><span class="action-pct" id="ap-DQN-0">0%</span></div>
            <div class="action-row"><span class="action-lbl">FIRE</span><div class="action-track"><div class="action-fill" id="ad-DQN-1" style="width:0%;background:var(--red);"></div></div><span class="action-pct" id="ap-DQN-1">0%</span></div>
            <div class="action-row"><span class="action-lbl">RIGHT</span><div class="action-track"><div class="action-fill" id="ad-DQN-2" style="width:0%;background:var(--blue);"></div></div><span class="action-pct" id="ap-DQN-2">0%</span></div>
            <div class="action-row"><span class="action-lbl">LEFT</span><div class="action-track"><div class="action-fill" id="ad-DQN-3" style="width:0%;background:var(--green);"></div></div><span class="action-pct" id="ap-DQN-3">0%</span></div>
        </div>
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
            <div class="card"><div class="lbl">Lives / Lost</div><div class="val" id="lives-Rainbow-Lite" style="font-size:16px;">5 / 0</div></div>
        </div>
        <div class="section">
            <div class="section-title">Q-Values</div>
            <div class="q-row"><span class="q-lbl">NOOP</span><div class="q-track"><div class="q-fill other" id="q-Rainbow-Lite-0"></div></div><span class="q-num" id="qv-Rainbow-Lite-0">0</span></div>
            <div class="q-row"><span class="q-lbl">FIRE</span><div class="q-track"><div class="q-fill other" id="q-Rainbow-Lite-1"></div></div><span class="q-num" id="qv-Rainbow-Lite-1">0</span></div>
            <div class="q-row"><span class="q-lbl">RIGHT</span><div class="q-track"><div class="q-fill other" id="q-Rainbow-Lite-2"></div></div><span class="q-num" id="qv-Rainbow-Lite-2">0</span></div>
            <div class="q-row"><span class="q-lbl">LEFT</span><div class="q-track"><div class="q-fill other" id="q-Rainbow-Lite-3"></div></div><span class="q-num" id="qv-Rainbow-Lite-3">0</span></div>
        </div>
        <div class="section">
            <div class="section-title">Action Distribution (this episode)</div>
            <div class="action-row"><span class="action-lbl">NOOP</span><div class="action-track"><div class="action-fill" id="ad-Rainbow-Lite-0" style="width:0%;background:var(--dim);"></div></div><span class="action-pct" id="ap-Rainbow-Lite-0">0%</span></div>
            <div class="action-row"><span class="action-lbl">FIRE</span><div class="action-track"><div class="action-fill" id="ad-Rainbow-Lite-1" style="width:0%;background:var(--red);"></div></div><span class="action-pct" id="ap-Rainbow-Lite-1">0%</span></div>
            <div class="action-row"><span class="action-lbl">RIGHT</span><div class="action-track"><div class="action-fill" id="ad-Rainbow-Lite-2" style="width:0%;background:var(--blue);"></div></div><span class="action-pct" id="ap-Rainbow-Lite-2">0%</span></div>
            <div class="action-row"><span class="action-lbl">LEFT</span><div class="action-track"><div class="action-fill" id="ad-Rainbow-Lite-3" style="width:0%;background:var(--green);"></div></div><span class="action-pct" id="ap-Rainbow-Lite-3">0%</span></div>
        </div>
        <div class="ftr">
            <span>Dueling CNN: V(s) + A(s,a)</span>
            <span>Replay: PER (alpha=0.6)</span>
            <span>Target: argmax(Q_online)</span>
        </div>
    </div>
</div>

<!-- Combined episode log spanning full width -->
<div class="combined-log">
    <div class="section-title" style="padding:8px 16px 4px;">Episode Log</div>
    <div class="combined-log-wrap">
        <table class="ep-table" id="combined-elog">
            <thead><tr><th>Agent</th><th>#</th><th>Score</th><th>Steps</th><th>Lost</th></tr></thead>
            <tbody id="combined-elog-body"><tr><td colspan="5" class="ep-log-empty">Waiting for first episode to complete...</td></tr></tbody>
        </table>
    </div>
</div>

<script>
const agents = ['DQN', 'Rainbow-Lite'];

function setSpeed(s) {
    fetch('/speed/' + s, {method:'POST'});
    document.querySelectorAll('.speed-btn').forEach(b => {
        b.classList.toggle('active', b.textContent === s + 'x');
    });
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

    const lives = document.getElementById('lives-' + name);
    if (lives && d.lives !== undefined) {
        const l = d.lives;
        const bl = d.balls_lost || 0;
        lives.textContent = l + ' / ' + bl;
        lives.style.color = l >= 4 ? 'var(--green)' : l >= 2 ? 'var(--orange)' : 'var(--red)';
    }

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

    // Action distribution
    if (d.action_counts && d.action_counts.length === 4) {
        const ac = d.action_counts;
        const total = ac.reduce((a, b) => a + b, 0) || 1;
        for (let i = 0; i < 4; i++) {
            const pct = (ac[i] / total * 100);
            const bar = document.getElementById('ad-' + name + '-' + i);
            if (bar) bar.style.width = pct + '%';
            const lbl = document.getElementById('ap-' + name + '-' + i);
            if (lbl) lbl.textContent = pct.toFixed(0) + '%';
        }
    }
}

// Combined episode log
const allLogs = [];  // [{agent, ep, score, steps, balls_lost, _seq}]
let logSeq = 0;
const lastLogLen = {'DQN': 0, 'Rainbow-Lite': 0};

function updateCombinedLog(data) {
    let changed = false;
    for (const name of agents) {
        const d = data[name];
        if (!d || !d.episode_log) continue;
        const log = d.episode_log;
        if (log.length > lastLogLen[name]) {
            // New episodes completed
            for (let i = lastLogLen[name]; i < log.length; i++) {
                allLogs.push({
                    agent: name,
                    ep: log[i].ep,
                    score: log[i].score,
                    steps: log[i].steps,
                    balls_lost: log[i].balls_lost,
                    _seq: logSeq++
                });
            }
            lastLogLen[name] = log.length;
            changed = true;
        }
    }
    if (!changed) return;

    // Sort by sequence descending (most recent first), keep last 20
    const rows = allLogs.slice(-20).reverse();
    const tbody = document.getElementById('combined-elog-body');
    if (!tbody) return;
    tbody.innerHTML = rows.map(r => {
        const cls = r.agent === 'DQN' ? 'ep-row-dqn' : 'ep-row-rl';
        return '<tr class="' + cls + '"><td>' + r.agent + '</td><td>' + r.ep + '</td><td>' + r.score + '</td><td>' + r.steps + '</td><td>' + r.balls_lost + '</td></tr>';
    }).join('');
}

const src = new EventSource('/stream');
src.onmessage = function(e) {
    const data = JSON.parse(e.data);
    for (const name of agents) {
        if (data[name]) updateAgent(name, data[name]);
    }
    updateCombinedLog(data);
};
</script>
</body>
</html>"""
