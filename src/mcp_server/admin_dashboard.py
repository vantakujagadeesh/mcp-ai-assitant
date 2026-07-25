"""
Admin Dashboard

Self-contained web-based admin dashboard with live metrics,
session viewer, cache controls, and escalation status.

Served as a single-page HTML application with embedded CSS and JS
(no external dependencies or build tools required).
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dashboard HTML template
# ---------------------------------------------------------------------------
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MCP AI Assistant — Admin Dashboard</title>
<style>
  /* ── Reset & Base ─────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg-primary: #0b0e14;
    --bg-secondary: #131821;
    --bg-card: #1a1f2e;
    --bg-card-hover: #1f2638;
    --border: #2a3142;
    --text-primary: #e4e8f1;
    --text-secondary: #8b95a8;
    --text-muted: #5a6478;
    --accent-blue: #4b8cff;
    --accent-green: #34d399;
    --accent-red: #f87171;
    --accent-yellow: #fbbf24;
    --accent-purple: #a78bfa;
    --accent-cyan: #22d3ee;
    --accent-orange: #fb923c;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow: 0 4px 24px rgba(0,0,0,0.3);
  }
  html { font-size: 14px; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.6;
  }

  /* ── Layout ──────────────────────────────────────── */
  .app { display: flex; min-height: 100vh; }

  /* ── Sidebar ─────────────────────────────────────── */
  .sidebar {
    width: 240px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    padding: 24px 16px;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }
  .sidebar-logo {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--accent-blue);
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .sidebar-logo svg { width: 24px; height: 24px; }
  .sidebar-nav { display: flex; flex-direction: column; gap: 4px; }
  .sidebar-nav a {
    padding: 10px 14px;
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
  }
  .sidebar-nav a:hover { background: var(--bg-card); color: var(--text-primary); }
  .sidebar-nav a.active { background: var(--accent-blue); color: #fff; }
  .sidebar-nav a .icon { width: 18px; height: 18px; opacity: 0.7; }
  .sidebar-nav a.active .icon { opacity: 1; }
  .sidebar-footer {
    margin-top: auto;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 0.75rem;
    color: var(--text-muted);
  }
  #server-time { color: var(--text-secondary); }

  /* ── Main Content ───────────────────────────────── */
  .main {
    flex: 1;
    padding: 28px 32px;
    overflow-y: auto;
    max-width: 1400px;
  }
  .main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 28px;
  }
  .main-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
  }
  .status-dot.online { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
  .status-dot.offline { background: var(--accent-red); box-shadow: 0 0 8px var(--accent-red); }
  .status-dot.warning { background: var(--accent-yellow); box-shadow: 0 0 8px var(--accent-yellow); }

  /* ── Metrics Grid ───────────────────────────────── */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }
  .metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    opacity: 0;
    transition: opacity 0.3s;
  }
  .metric-card:hover::before { opacity: 1; }
  .metric-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--accent-blue);
    transform: translateY(-2px);
    box-shadow: var(--shadow);
  }
  .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.2;
  }
  .metric-value.green { color: var(--accent-green); }
  .metric-value.blue { color: var(--accent-blue); }
  .metric-value.yellow { color: var(--accent-yellow); }
  .metric-value.red { color: var(--accent-red); }
  .metric-value.purple { color: var(--accent-purple); }
  .metric-value.cyan { color: var(--accent-cyan); }
  .metric-value.orange { color: var(--accent-orange); }
  .metric-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
  }

  /* ── Section ─────────────────────────────────────── */
  .section { margin-bottom: 28px; }
  .section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-title .badge {
    background: var(--bg-card);
    color: var(--text-secondary);
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
    font-weight: 500;
  }

  /* ── Table ──────────────────────────────────────── */
  .table-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th {
    background: var(--bg-secondary);
    padding: 12px 16px;
    text-align: left;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 12px 16px;
    font-size: 0.88rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--bg-card-hover); }
  td .mono { font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; font-size: 0.8rem; }
  .pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
  }
  .pill.green { background: rgba(52, 211, 153, 0.15); color: var(--accent-green); }
  .pill.red { background: rgba(248, 113, 113, 0.15); color: var(--accent-red); }
  .pill.yellow { background: rgba(251, 191, 36, 0.15); color: var(--accent-yellow); }
  .pill.blue { background: rgba(75, 140, 255, 0.15); color: var(--accent-blue); }
  .pill.purple { background: rgba(167, 139, 250, 0.15); color: var(--accent-purple); }

  /* ── Buttons ────────────────────────────────────── */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-card);
    color: var(--text-primary);
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn:hover { background: var(--bg-card-hover); border-color: var(--accent-blue); }
  .btn.danger:hover { border-color: var(--accent-red); color: var(--accent-red); }
  .btn.primary { background: var(--accent-blue); border-color: var(--accent-blue); color: #fff; }
  .btn.primary:hover { background: #3a7be8; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-sm { padding: 4px 10px; font-size: 0.75rem; }

  /* ── Chat bubble (session viewer) ────────────────── */
  .chat { padding: 16px; display: flex; flex-direction: column; gap: 10px; max-height: 400px; overflow-y: auto; }
  .chat-msg {
    max-width: 80%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 0.85rem;
    line-height: 1.5;
  }
  .chat-msg.user {
    align-self: flex-end;
    background: var(--accent-blue);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .chat-msg.assistant {
    align-self: flex-start;
    background: var(--bg-secondary);
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
  }
  .chat-msg .meta {
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-top: 4px;
  }
  .chat-msg.user .meta { color: rgba(255,255,255,0.6); }

  /* ── Toast notifications ────────────────────────── */
  .toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .toast {
    padding: 12px 20px;
    border-radius: var(--radius-sm);
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-primary);
    font-size: 0.85rem;
    box-shadow: var(--shadow);
    animation: slideIn 0.3s ease;
    min-width: 250px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .toast.success { border-left: 3px solid var(--accent-green); }
  .toast.error { border-left: 3px solid var(--accent-red); }
  .toast.info { border-left: 3px solid var(--accent-blue); }
  @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

  /* ── Modal ──────────────────────────────────────── */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    animation: fadeIn 0.2s;
  }
  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    width: 90%;
    max-width: 640px;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: var(--shadow);
  }
  .modal h2 { margin-bottom: 16px; font-size: 1.2rem; }
  .modal-actions { display: flex; gap: 8px; margin-top: 16px; justify-content: flex-end; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  /* ── Loader ─────────────────────────────────────── */
  .loader {
    display: inline-block;
    width: 20px; height: 20px;
    border: 2px solid var(--border);
    border-top-color: var(--accent-blue);
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Empty state ────────────────────────────────── */
  .empty { text-align: center; padding: 40px 20px; color: var(--text-muted); }

  /* ── Responsive ─────────────────────────────────── */
  @media (max-width: 900px) {
    .sidebar { width: 60px; padding: 16px 8px; }
    .sidebar .nav-label { display: none; }
    .sidebar-logo span { display: none; }
    .main { padding: 20px 16px; }
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
  }

  /* ── Scrollbar ──────────────────────────────────── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
</head>
<body>
<div class="app">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-logo">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
      <span>MCP Admin</span>
    </div>
    <nav class="sidebar-nav">
      <a class="active" onclick="switchTab('overview')">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
        <span class="nav-label">Overview</span>
      </a>
      <a onclick="switchTab('sessions')">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span class="nav-label">Sessions</span>
      </a>
      <a onclick="switchTab('cache')">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7V4h16v3"/><path d="M4 11V9h16v2"/><path d="M4 15v-2h16v2"/><path d="M4 19v-2h16v2"/></svg>
        <span class="nav-label">Cache</span>
      </a>
      <a onclick="switchTab('escalation')">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
        <span class="nav-label">Escalation</span>
      </a>
      <a onclick="switchTab('feedback')">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
        <span class="nav-label">Feedback</span>
      </a>
      <a onclick="switchTab('eval')">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/><circle cx="12" cy="12" r="2"/></svg>
        <span class="nav-label">RAG Eval</span>
      </a>
      <a onclick="switchTab('monitoring')">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        <span class="nav-label">Monitoring</span>
      </a>
    </nav>
    <div class="sidebar-footer">
      <div id="server-time">—</div>
      <div style="margin-top:4px">v2.0.0</div>
    </div>
  </aside>

  <!-- Main -->
  <main class="main">
    <div class="main-header">
      <h1>Admin Dashboard</h1>
      <div>
        <span class="status-dot online" id="status-dot"></span>
        <span id="status-text" style="font-size:0.85rem;color:var(--text-secondary)">Connected</span>
        <button class="btn btn-sm" onclick="refreshAll()" style="margin-left:12px;">⟳ Refresh</button>
      </div>
    </div>

    <!-- Overview Tab -->
    <div id="tab-overview" class="tab-content">
      <div class="metrics-grid" id="metrics-grid"></div>

      <div class="section">
        <div class="section-title">System Health</div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Component</th><th>Status</th><th>Details</th></tr></thead>
            <tbody id="health-table"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Sessions Tab -->
    <div id="tab-sessions" class="tab-content" style="display:none">
      <div class="section">
        <div class="section-title">
          Active Sessions <span class="badge" id="session-count">0</span>
          <button class="btn btn-sm" onclick="refreshSessions()" style="margin-left:auto;">⟳ Refresh</button>
        </div>
        <div class="table-wrap" id="sessions-container">
          <div class="empty"><div class="loader"></div><p style="margin-top:8px">Loading sessions…</p></div>
        </div>
      </div>
    </div>

    <!-- Cache Tab -->
    <div id="tab-cache" class="tab-content" style="display:none">
      <div class="section">
        <div class="section-title">
          Cache Statistics
          <button class="btn btn-sm" onclick="refreshCache()" style="margin-left:auto;">⟳ Refresh</button>
          <button class="btn btn-sm danger" onclick="clearCache()" style="margin-left:8px;">🗑 Clear Cache</button>
        </div>
        <div class="metrics-grid" id="cache-metrics"></div>
      </div>
      <div class="section">
        <div class="section-title">Cached Keys</div>
        <div class="table-wrap">
          <table><thead><tr><th>Key</th><th>TTL</th></tr></thead><tbody id="cache-keys"></tbody></table>
        </div>
      </div>
    </div>

    <!-- Escalation Tab -->
    <div id="tab-escalation" class="tab-content" style="display:none">
      <div class="section">
        <div class="section-title">n8n Escalation Status</div>
        <div class="metrics-grid" id="escalation-metrics"></div>
      </div>
    </div>

    <!-- Feedback Tab -->
    <div id="tab-feedback" class="tab-content" style="display:none">
      <div class="section">
        <div class="section-title">
          Feedback Log
          <button class="btn btn-sm" onclick="refreshFeedback()" style="margin-left:auto;">⟳ Refresh</button>
        </div>
        <div class="table-wrap" id="feedback-container">
          <div class="empty"><div class="loader"></div><p style="margin-top:8px">Loading feedback…</p></div>
        </div>
      </div>
    </div>

    <!-- RAG Eval Tab -->
    <div id="tab-eval" class="tab-content" style="display:none">
      <div class="section">
        <div class="section-title">
          RAG Evaluation Scores
          <span class="badge" id="eval-engine">—</span>
          <button class="btn btn-sm" onclick="refreshEval()" style="margin-left:auto;">⟳ Refresh</button>
        </div>
        <div class="metrics-grid" id="eval-metrics"></div>
      </div>
      <div class="section">
        <div class="section-title">Per-Question Breakdown</div>
        <div class="table-wrap">
          <table><thead><tr><th>#</th><th>Question</th><th>Faithfulness</th><th>Answer Rel.</th><th>Context Rec.</th><th>Context Prec.</th></tr></thead><tbody id="eval-samples"></tbody></table>
        </div>
      </div>
    </div>

    <!-- Monitoring Tab (NEW: Prometheus + Grafana) -->
    <div id="tab-monitoring" class="tab-content" style="display:none">
      <div class="section">
        <div class="section-title">
          Prometheus Metrics
          <button class="btn btn-sm" onclick="refreshMonitoring()" style="margin-left:auto;">⟳ Refresh</button>
          <a href="/metrics" target="_blank" class="btn btn-sm" style="margin-left:8px;text-decoration:none">📊 Raw Metrics</a>
        </div>
        <div class="metrics-grid" id="prometheus-metrics"></div>
      </div>
      <div class="section">
        <div class="section-title">
          Grafana Dashboard
          <a href="http://localhost:3000/d/mcp-ai-assistant-monitoring/mcp-ai-assistant-monitoring" target="_blank" class="btn btn-sm primary" style="margin-left:8px;text-decoration:none">📈 Open Grafana</a>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Service</th><th>URL</th><th>Status</th></tr></thead>
            <tbody id="monitoring-services"></tbody>
          </table>
        </div>
      </div>
    </div>
  </main>
</div>

<!-- Modal -->
<div id="modal" class="modal-overlay" style="display:none" onclick="if(event.target===this)closeModal()">
  <div class="modal" id="modal-content"></div>
</div>

<!-- Toast -->
<div class="toast-container" id="toasts"></div>

<script>
// ── State ──────────────────────────────────────────────────
let metrics = {};
let sessions = [];
let refreshInterval = null;
const API = '';  // same-origin

// Get or prompt for API key (stored in localStorage)
function getApiKey() {
  let key = localStorage.getItem('mcp_api_key');
  if (!key) {
    key = prompt('Enter your MCP API key (required for admin actions):');
    if (key) localStorage.setItem('mcp_api_key', key);
  }
  return key;
}

function authHeaders() {
  const key = getApiKey();
  return key ? { 'X-API-Key': key, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
}

function clearStoredKey() {
  localStorage.removeItem('mcp_api_key');
  toast('API key cleared', 'info');
}

// ── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  refreshAll();
  refreshInterval = setInterval(refreshMetrics, 5000);
  updateClock();
  setInterval(updateClock, 1000);
});

// ── Tabs ──────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
  document.getElementById(`tab-${name}`).style.display = 'block';
  document.querySelectorAll('.sidebar-nav a').forEach(a => a.classList.remove('active'));
  document.querySelectorAll('.sidebar-nav a').forEach(a => {
    if (a.textContent.trim().toLowerCase().includes(name)) a.classList.add('active');
  });
  if (name === 'sessions') refreshSessions();
  if (name === 'cache') refreshCache();
  if (name === 'feedback') refreshFeedback();
  if (name === 'eval') refreshEval();
  if (name === 'monitoring') refreshMonitoring();
}

// ── Clock ─────────────────────────────────────────────────
function updateClock() {
  document.getElementById('server-time').textContent = new Date().toLocaleTimeString();
}

// ── Toast ─────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const c = document.getElementById('toasts');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icons = { success: '✓', error: '✗', info: 'ℹ' };
  el.innerHTML = `${icons[type] || 'ℹ'} ${msg}`;
  c.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3000);
}

// ── Modal ─────────────────────────────────────────────────
function openModal(html) { document.getElementById('modal-content').innerHTML = html; document.getElementById('modal').style.display = 'flex'; }
function closeModal() { document.getElementById('modal').style.display = 'none'; }

// ── API call ──────────────────────────────────────────────
async function api(path, opts = {}) {
  try {
    const headers = { 'Content-Type': 'application/json', ...opts.headers };
    const resp = await fetch(`${API}${path}`, { headers, ...opts });
    if (resp.status === 401 || resp.status === 403) {
      // Auth error — maybe the key is wrong, prompt again
      localStorage.removeItem('mcp_api_key');
      const newKey = getApiKey();
      if (newKey) {
        headers['X-API-Key'] = newKey;
        const retry = await fetch(`${API}${path}`, { headers, ...opts });
        if (retry.ok) return await retry.json();
        const e2 = await retry.json().catch(() => ({}));
        throw new Error(e2.detail || e2.message || `HTTP ${retry.status}`);
      }
    }
    if (!resp.ok) { const e = await resp.json().catch(() => ({})); throw new Error(e.detail || e.message || `HTTP ${resp.status}`); }
    return await resp.json();
  } catch (e) { toast(e.message, 'error'); throw e; }
}

// Authenticated API call (includes API key in headers)
async function apiAuthed(path, opts = {}) {
  const key = getApiKey();
  if (!key) { toast('API key required for this action', 'error'); throw new Error('No API key'); }
  return api(path, { ...opts, headers: { ...opts.headers, 'X-API-Key': key } });
}

// ── Refresh All ──────────────────────────────────────────
async function refreshAll() {
  await Promise.all([refreshMetrics(), refreshSessions(), refreshCache(), refreshFeedback(), refreshEval(), refreshMonitoring()]);
}

async function refreshMetrics() {
  try {
    const h = await api('/health');
    metrics.health = h;
    renderMetrics(h);
    renderHealthTable(h);
    if (h.n8n_configured !== undefined) renderEscalation(h);
  } catch (e) { /* toast already shown */ }
}

async function refreshSessions() {
  try {
    const data = await api('/api/sessions?limit=50');
    sessions = data.sessions || [];
    renderSessions(sessions);
  } catch (e) {}
}

async function refreshCache() {
  try {
    const data = await api('/api/cache/stats');
    renderCacheStats(data);
  } catch (e) {}
}

async function refreshFeedback() {
  try {
    // Feedback is stored as JSONL — fetch via a simple endpoint would need a dedicated API
    document.getElementById('feedback-container').innerHTML =
      '<div class="empty"><p>Feedback log viewer requires a dedicated endpoint. Data stored in <code>data/feedback/feedback_log.jsonl</code></p>' +
      '<button class="btn btn-sm primary" onclick="toast(\'Tip: Use the /api/feedback_log endpoint to submit feedback\', \'info\')" style="margin-top:12px;">How to submit feedback</button></div>';
  } catch (e) {}
}

// ── Render Metrics ───────────────────────────────────────
function renderMetrics(h) {
  const grid = document.getElementById('metrics-grid');
  const cards = [
    { label: 'Status', value: h.status || 'ok', cls: 'green', sub: `v${h.version}` },
    { label: 'Model', value: h.model_loaded ? 'Loaded' : 'Offline', cls: h.model_loaded ? 'green' : 'red', sub: h.model_loaded ? 'LLM ready' : 'Fallback mode' },
    { label: 'KB Articles', value: h.rag_articles || 0, cls: 'blue', sub: h.rag_loaded ? 'RAG active' : 'RAG offline' },
    { label: 'Vector DB', value: h.vector_db?.connected ? '🚀 Qdrant' : '💾 In-Memory', cls: h.vector_db?.connected ? 'green' : 'yellow', sub: h.vector_db ? `${h.vector_db.point_count} points` : '—' },
    { label: 'Sessions', value: h.conversation_sessions || 0, cls: 'purple', sub: 'Active conversations' },
    { label: 'DPO Training', value: h.dpo_training?.total_pairs_available >= (h.dpo_training?.min_pairs_for_training || 50) ? '🎯 Ready' : '⏳ Collecting', cls: h.dpo_training?.total_pairs_available >= (h.dpo_training?.min_pairs_for_training || 50) ? 'green' : 'yellow', sub: `${h.dpo_training?.total_pairs_available || 0} pairs (need ${h.dpo_training?.min_pairs_for_training || 50})` },
    { label: 'PII Guard', value: h.guardrails?.pii?.engine === 'presidio' ? '🛡️ Presidio' : '🛡️ Regex', cls: h.guardrails?.pii?.enabled ? 'green' : 'yellow', sub: h.guardrails?.pii?.enabled ? 'Active' : 'Disabled' },
    { label: 'Safety Guard', value: h.guardrails?.safety?.engine === 'nemo' ? '🛡️ NeMo' : '🛡️ Heuristic', cls: h.guardrails?.safety?.enabled ? 'green' : 'yellow', sub: h.guardrails?.safety?.enabled ? 'Active' : 'Disabled' },
    { label: 'RAG Eval', value: h.rag_eval_available ? '📊 ' + (h.rag_eval_engine || '—') : '📊 Not run', cls: h.rag_eval_available ? 'green' : 'yellow', sub: h.rag_eval_available ? `F: ${h.rag_eval_faithfulness?.toFixed(3) || '—'}, CR: ${h.rag_eval_context_recall?.toFixed(3) || '—'}` : 'Run: python src/eval/ragas_eval.py' },
    { label: 'Auth', value: h.auth_enabled ? '🔒 On' : '🔓 Off', cls: h.auth_enabled ? 'green' : 'yellow', sub: 'API key auth' },
    { label: 'Rate Limit', value: h.rate_limiting_enabled ? '⏱ On' : '⏱ Off', cls: h.rate_limiting_enabled ? 'cyan' : 'orange', sub: 'Sliding window' },
    { label: 'n8n Escalation', value: h.n8n_configured ? 'Connected' : 'Not configured', cls: h.n8n_configured ? 'green' : 'red', sub: h.n8n_base_url || '—' },
  ];
  grid.innerHTML = cards.map(c => `
    <div class="metric-card">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}">${c.value}</div>
      <div class="metric-sub">${c.sub}</div>
    </div>
  `).join('');
}

// ── Render Health Table ──────────────────────────────────
function renderHealthTable(h) {
  const rows = [
    { comp: 'RAG Retriever', status: h.rag_loaded ? '✅ Online' : '❌ Offline', detail: `${h.rag_articles} articles` },
    { comp: 'ML Model', status: h.model_loaded ? '✅ Loaded' : '⚠️ Fallback', detail: h.model_loaded ? 'LLM ready' : 'Keyword-based' },
    { comp: 'Auth Middleware', status: h.auth_enabled ? '✅ Active' : '⚠️ Disabled', detail: 'API key validation' },
    { comp: 'Rate Limiting', status: h.rate_limiting_enabled ? '✅ Active' : '⚠️ Disabled', detail: 'Per-key sliding window' },
    { comp: 'Vector DB', status: h.vector_db?.connected ? '✅ Qdrant' : '💾 In-Memory', detail: h.vector_db ? `${h.vector_db.type} · ${h.vector_db.point_count || 0} points` : 'Not available' },
    { comp: 'PII Masking', status: h.guardrails?.pii?.enabled ? `✅ ${h.guardrails.pii.engine}` : '⚠️ Disabled', detail: h.guardrails?.pii?.enabled ? `Engine: ${h.guardrails.pii.engine}` : 'Set guardrails.pii.enabled=true' },
    { comp: 'Safety Guard', status: h.guardrails?.safety?.enabled ? `✅ ${h.guardrails.safety.engine}` : '⚠️ Disabled', detail: h.guardrails?.safety?.enabled ? `Engine: ${h.guardrails.safety.engine}` : 'Set guardrails.safety.enabled=true' },
    { comp: 'DPO Pipeline', status: h.dpo_training?.enabled ? `✅ ${h.dpo_training.total_pairs_available || 0} pairs` : '⚠️ Disabled', detail: h.dpo_training?.total_pairs_available >= (h.dpo_training?.min_pairs_for_training || 50) ? `🎯 Ready for training (${h.dpo_training.total_training_runs || 0} runs so far)` : `Collecting pairs: ${h.dpo_training?.total_pairs_available || 0}/${h.dpo_training?.min_pairs_for_training || 50}` },
    { comp: 'RAG Evaluation', status: h.rag_eval_available ? `✅ ${h.rag_eval_num_samples || 0} samples` : '⚠️ Not run', detail: h.rag_eval_available ? `F: ${(h.rag_eval_faithfulness || 0).toFixed(3)} | AR: ${(h.rag_eval_answer_relevancy || 0).toFixed(3)} | CR: ${(h.rag_eval_context_recall || 0).toFixed(3)} | CP: ${(h.rag_eval_context_precision || 0).toFixed(3)}` : 'Run: python src/eval/ragas_eval.py' },
    { comp: 'n8n Webhook', status: h.n8n_configured ? '✅ Configured' : '⚠️ Not configured', detail: h.n8n_base_url || 'Set n8n.enabled=true' },
  ];
  document.getElementById('health-table').innerHTML = rows.map(r => `
    <tr><td>${r.comp}</td><td><span class="pill ${r.status.includes('✅') ? 'green' : 'yellow'}">${r.status.replace(/[✅⚠️]/g,'').trim()}</span></td><td>${r.detail}</td></tr>
  `).join('');
}

// ── Render Sessions ──────────────────────────────────────
function renderSessions(sessions) {
  document.getElementById('session-count').textContent = sessions.length;
  const container = document.getElementById('sessions-container');
  if (!sessions.length) {
    container.innerHTML = '<div class="empty"><p style="font-size:1.2rem;margin-bottom:8px;">💬 No active sessions</p><p>Start a conversation via <code>POST /api/conversation/message</code> to see it here.</p></div>';
    return;
  }
  let html = '<table><thead><tr><th>Session ID</th><th>Messages</th><th>Last Active</th><th>Actions</th></tr></thead><tbody>';
  sessions.forEach(s => {
    const time = new Date(s.updated_at * 1000).toLocaleString();
    html += `<tr>
      <td><span class="mono">${s.session_id}</span></td>
      <td>${s.message_count}</td>
      <td>${time}</td>
      <td><button class="btn btn-sm" onclick="viewSession('${s.session_id}')">👁 View</button> <button class="btn btn-sm danger" onclick="deleteSession('${s.session_id}')">🗑</button></td>
    </tr>`;
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

async function viewSession(id) {
  try {
    const data = await api(`/api/conversation/${id}?max_messages=50`);
    let html = `<h2>Session: <span class="mono" style="font-size:0.85rem">${id}</span></h2>
      <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:12px">${data.message_count} messages</div>
      <div class="chat">`;
    data.messages.forEach(m => {
      const time = new Date(m.timestamp * 1000).toLocaleString();
      html += `<div class="chat-msg ${m.role}"><div>${m.content}</div><div class="meta">${m.role} · ${time}</div></div>`;
    });
    html += '</div><div class="modal-actions"><button class="btn btn-sm danger" onclick="deleteSession(\''+id+'\');closeModal();">Delete Session</button><button class="btn btn-sm" onclick="closeModal()">Close</button></div>';
    openModal(html);
  } catch (e) {}
}

async function deleteSession(id) {
  if (!confirm(`Delete session "${id}"?`)) return;
  try {
    await api(`/api/sessions/${id}`, { method: 'DELETE' });
    toast(`Session ${id} deleted`, 'success');
    refreshSessions();
  } catch (e) {}
}

// ── Render Cache ─────────────────────────────────────────
function renderCacheStats(data) {
  const s = data.stats || {};
  const cards = [
    { label: 'Cache Type', value: data.cache_type || '—', cls: 'blue', sub: '' },
    { label: 'Size', value: s.size || 0, cls: 'purple', sub: `max ${s.max_size || '—'}` },
    { label: 'Hit Rate', value: `${((s.hit_rate || 0) * 100).toFixed(1)}%`, cls: s.hit_rate > 0.5 ? 'green' : 'yellow', sub: `${s.hits || 0} hits` },
    { label: 'Misses', value: s.misses || 0, cls: 'red', sub: '' },
  ];
  document.getElementById('cache-metrics').innerHTML = cards.map(c => `
    <div class="metric-card">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}">${c.value}</div>
      <div class="metric-sub">${c.sub}</div>
    </div>
  `).join('');

  if (s.keys) {
    document.getElementById('cache-keys').innerHTML = s.keys.map(k =>
      `<tr><td class="mono">${k}</td><td>—</td></tr>`
    ).join('') || '<tr><td colspan="2" class="empty">No keys in cache</td></tr>';
  }
}

async function clearCache() {
  if (!confirm('Clear the entire cache?')) return;
  try {
    const data = await apiAuthed('/api/cache/clear', { method: 'POST' });
    toast('Cache cleared', 'success');
    refreshCache();
  } catch (e) { toast('Failed to clear cache: ' + e.message, 'error'); }
}

// ── Monitoring (Prometheus) ─────────────────────────────
async function refreshMonitoring() {
  const grid = document.getElementById('prometheus-metrics');
  const tbody = document.getElementById('monitoring-services');

  // Show Prometheus status from health endpoint
  let metricsEnabled = false;
  let metricsEndpoint = null;
  try {
    const h = await api('/health');
    metricsEnabled = h.metrics_enabled || false;
    metricsEndpoint = h.metrics_endpoint || null;
  } catch (e) {}

  const cards = [
    { label: 'Prometheus', value: metricsEnabled ? '✅ Enabled' : '⚠️ Disabled', cls: metricsEnabled ? 'green' : 'red', sub: metricsEndpoint || '/metrics' },
    { label: 'Metrics Endpoint', value: '/metrics', cls: 'blue', sub: 'Prometheus text format' },
    { label: 'Scrape Interval', value: '15s', cls: 'cyan', sub: 'Configured in prometheus.yml' },
    { label: 'Retention', value: '30d', cls: 'purple', sub: 'Prometheus TSDB' },
  ];
  grid.innerHTML = cards.map(c => `
    <div class="metric-card">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}">${c.value}</div>
      <div class="metric-sub">${c.sub}</div>
    </div>
  `).join('');

  // Services table
  const services = [
    { svc: 'MCP API', url: 'http://localhost:8000/metrics', status: '⏳' },
    { svc: 'Prometheus', url: 'http://localhost:9090', status: '⏳' },
    { svc: 'Grafana', url: 'http://localhost:3000', status: '⏳' },
    { svc: 'Qdrant', url: 'http://localhost:6333/healthz', status: '⏳' },
  ];

  // Try to check each service by fetching it
  for (const s of services) {
    try {
      const resp = await fetch(s.url, { method: 'HEAD', mode: 'no-cors' });
      s.status = '✅ Reachable';
    } catch (e) {
      s.status = '⚠️ Unreachable (start docker compose up)';
    }
  }

  tbody.innerHTML = services.map(s => `
    <tr><td>${s.svc}</td><td><code style="color:var(--accent-cyan)">${s.url}</code></td><td><span class="pill ${s.status.includes('✅') ? 'green' : 'yellow'}">${s.status}</span></td></tr>
  `).join('');
}

// ── RAG Eval ────────────────────────────────────────────
async function refreshEval() {
  try {
    const data = await api('/api/eval/ragas');
    renderEval(data);
  } catch (e) { }
}

function renderEval(data) {
  document.getElementById('eval-engine').textContent = data.engine || 'N/A';

  const grid = document.getElementById('eval-metrics');
  if (!data.available) {
    grid.innerHTML = '<div class="empty"><p style="font-size:1.2rem;margin-bottom:8px;">📊 No Evaluation Runs</p><p>Run <code>python src/eval/ragas_eval.py</code> to generate scores.</p></div>';
    document.getElementById('eval-samples').innerHTML = '';
    return;
  }

  const cards = [
    { label: 'Faithfulness', value: (data.faithfulness || 0).toFixed(4), cls: (data.faithfulness || 0) > 0.7 ? 'green' : 'yellow', sub: data.deltas?.faithfulness !== undefined ? `${data.deltas.faithfulness >= 0 ? '▲' : '▼'} ${(data.deltas.faithfulness || 0).toFixed(4)}` : '' },
    { label: 'Answer Relevancy', value: (data.answer_relevancy || 0).toFixed(4), cls: (data.answer_relevancy || 0) > 0.5 ? 'green' : 'yellow', sub: data.deltas?.answer_relevancy !== undefined ? `${data.deltas.answer_relevancy >= 0 ? '▲' : '▼'} ${(data.deltas.answer_relevancy || 0).toFixed(4)}` : '' },
    { label: 'Context Recall', value: (data.context_recall || 0).toFixed(4), cls: (data.context_recall || 0) > 0.7 ? 'green' : 'yellow', sub: data.deltas?.context_recall !== undefined ? `${data.deltas.context_recall >= 0 ? '▲' : '▼'} ${(data.deltas.context_recall || 0).toFixed(4)}` : '' },
    { label: 'Context Precision', value: (data.context_precision || 0).toFixed(4), cls: (data.context_precision || 0) > 0.5 ? 'green' : 'yellow', sub: data.deltas?.context_precision !== undefined ? `${data.deltas.context_precision >= 0 ? '▲' : '▼'} ${(data.deltas.context_precision || 0).toFixed(4)}` : '' },
    { label: 'Engine', value: data.engine || 'heuristic', cls: 'blue', sub: `${data.num_samples || 0} samples` },
    { label: 'Total Runs', value: data.total_runs || 0, cls: 'purple', sub: data.timestamp ? new Date(data.timestamp).toLocaleString() : '' },
  ];
  grid.innerHTML = cards.map(c => `
    <div class="metric-card">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}">${c.value}</div>
      <div class="metric-sub">${c.sub}</div>
    </div>
  `).join('');

  // Per-sample table
  const samples = data.samples || [];
  const tbody = document.getElementById('eval-samples');
  if (!samples.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">No sample data available</td></tr>';
    return;
  }
  tbody.innerHTML = samples.map((s, i) => `
    <tr>
      <td>${i + 1}</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${s.question}">${s.question}</td>
      <td>${(s.faithfulness || 0).toFixed(3)}</td>
      <td>${(s.answer_relevancy || 0).toFixed(3)}</td>
      <td>${(s.context_recall || 0).toFixed(3)}</td>
      <td>${(s.context_precision || 0).toFixed(3)}</td>
    </tr>
  `).join('');
}

// ── Render Escalation ────────────────────────────────────
function renderEscalation(h) {
  const cards = [
    { label: 'n8n Status', value: h.n8n_configured ? '✅ Configured' : '⚠️ Not Configured', cls: h.n8n_configured ? 'green' : 'red', sub: '' },
    { label: 'Webhook URL', value: h.n8n_base_url || '—', cls: 'blue', sub: 'Base URL' },
  ];
  // If escalation status endpoint gives more detail, fetch it
  api('/api/escalate/status').then(d => {
    cards.push({ label: 'Webhook Path', value: d.webhook_path || '—', cls: 'purple', sub: '' });
    document.getElementById('escalation-metrics').innerHTML = cards.map(c => `
      <div class="metric-card">
        <div class="metric-label">${c.label}</div>
        <div class="metric-value ${c.cls}">${c.value}</div>
        <div class="metric-sub">${c.sub}</div>
      </div>
    `).join('');
  }).catch(() => {
    document.getElementById('escalation-metrics').innerHTML = cards.map(c => `
      <div class="metric-card">
        <div class="metric-label">${c.label}</div>
        <div class="metric-value ${c.cls}">${c.value}</div>
        <div class="metric-sub">${c.sub}</div>
      </div>
    `).join('');
  });
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Mount dashboard routes on a FastAPI app
# ---------------------------------------------------------------------------
def mount_dashboard(app: Any) -> None:
    """Mount admin dashboard routes on the provided FastAPI app.

    Adds:
    - ``GET /admin/dashboard`` → the HTML dashboard page
    - ``POST /api/cache/clear`` → clear the entire cache (admin)
    """

    from fastapi import Depends, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from starlette.responses import Response
    from src.mcp_server.auth import require_admin_key

    # ── GET /admin/dashboard ───────────────────────────────────────
    @app.get("/admin/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def admin_dashboard(request: Request):
        return HTMLResponse(content=DASHBOARD_HTML)

    # ── POST /api/cache/clear ──────────────────────────────────────
    @app.post("/api/cache/clear")
    async def admin_clear_cache(
        request: Request,
        admin_key: Dict[str, str] = Depends(require_admin_key),
    ):
        """Clear the entire cache. Requires admin API key."""
        # Import the module-level cache reference
        from src.mcp_server.rest_api import _cache

        if _cache is None:
            raise HTTPException(status_code=503, detail="Cache not initialised")

        try:
            import asyncio
            if asyncio.iscoroutinefunction(_cache.clear):
                await _cache.clear()
            else:
                _cache.clear()
            return {"status": "cleared", "cache_type": "Redis" if hasattr(_cache, "_redis") else "LRU (In-Memory)"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to clear cache: {exc}")

    logger.info("Admin dashboard mounted at /admin/dashboard")
    logger.info("Cache clear endpoint mounted at POST /api/cache/clear")
