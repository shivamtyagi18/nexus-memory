# Episode Feed Tab Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `📼 Episode Feed` tab to the Smriti Memory Browser UI that shows all raw episodes from `episodes.db` in reverse-chronological order with consolidation status.

**Architecture:** A new `_read_episodes()` function reads `episodes.db` via stdlib `sqlite3` and is exposed via a new `/api/episodes` GET endpoint. The browser tab lazily fetches this endpoint on first activation and caches the result until the global `↻ Refresh` button resets the `episodesLoaded` flag. All changes are confined to `smriti_memcore/ui/server.py` — the single-file embedded UI server — plus a new test file.

**Tech Stack:** Python stdlib (`sqlite3`, `json`, `http.server`), vanilla JS, existing CSS classes in the embedded HTML string.

**Spec:** `docs/superpowers/specs/2026-04-12-episode-feed-tab-design.md`

---

## File Map

| File | Change |
|------|--------|
| `smriti_memcore/ui/server.py` | Add `import sqlite3`, `_read_episodes()`, `/api/episodes` route, tab HTML+CSS, JS `buildEpisodes()`, update `showTab()` and `refreshData()` |
| `tests/test_ui_server.py` | New test file for `_read_episodes()` and `/api/episodes` endpoint |

---

## Task 1: `_read_episodes()` — data function

**Files:**
- Create: `tests/test_ui_server.py`
- Modify: `smriti_memcore/ui/server.py` (add `import sqlite3`; add `_read_episodes` after `_read_palace`)

- [ ] **Step 1: Write failing tests for `_read_episodes`**

Create `tests/test_ui_server.py`:

```python
"""Tests for smriti_memcore.ui.server — episode feed data layer."""
import json
import sqlite3
import pytest
from pathlib import Path
from smriti_memcore.ui.server import _read_episodes


def _make_db(tmp_path, rows):
    """Create a minimal episodes.db with given rows."""
    db = tmp_path / "episodes" / "episodes.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            content TEXT,
            timestamp TEXT,
            salience_json TEXT,
            source TEXT,
            trajectory_id TEXT,
            trajectory_step INTEGER,
            reflections_json TEXT,
            consolidated INTEGER DEFAULT 0,
            metadata_json TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return str(tmp_path)


def test_read_episodes_returns_list_of_dicts(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "content one", "2026-04-12T10:00:00", '{"composite": 0.5}',
         "user_stated", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert isinstance(result, list)
    assert len(result) == 1
    ep = result[0]
    assert ep["id"] == "id-1"
    assert ep["content"] == "content one"
    assert ep["timestamp"] == "2026-04-12T10:00:00"
    assert ep["source"] == "user_stated"
    assert ep["salience"] == pytest.approx(0.5)
    assert ep["consolidated"] is True  # must be Python bool, not int 1


def test_read_episodes_consolidated_cast_to_bool(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-0", "pending ep", "2026-04-12T09:00:00", '{"composite":0.1}',
         "direct", None, 0, None, 0, None),
        ("id-1", "done ep",   "2026-04-12T10:00:00", '{"composite":0.9}',
         "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    by_id = {ep["id"]: ep for ep in result}
    assert by_id["id-0"]["consolidated"] is False
    assert by_id["id-1"]["consolidated"] is True


def test_read_episodes_null_salience_json(tmp_path):
    # NULL salience_json → Python None. The guard `None or "{}"` evaluates to "{}"
    # before json.loads is called, so json.loads succeeds with an empty dict → 0.0.
    storage = _make_db(tmp_path, [
        ("id-1", "no sal", "2026-04-12T10:00:00", None,
         "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert result[0]["salience"] == pytest.approx(0.0)


def test_read_episodes_malformed_salience_json(tmp_path):
    # Malformed JSON → json.JSONDecodeError caught → default 0.0
    storage = _make_db(tmp_path, [
        ("id-1", "bad sal", "2026-04-12T10:00:00", "not-json",
         "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert result[0]["salience"] == pytest.approx(0.0)


def test_read_episodes_ordered_newest_first(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "old", "2026-04-10T10:00:00", '{"composite":0.3}', "direct", None, 0, None, 1, None),
        ("id-2", "new", "2026-04-12T10:00:00", '{"composite":0.7}', "direct", None, 0, None, 1, None),
        ("id-3", "mid", "2026-04-11T10:00:00", '{"composite":0.5}', "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert [ep["id"] for ep in result] == ["id-2", "id-3", "id-1"]


def test_read_episodes_missing_db_returns_empty(tmp_path):
    result = _read_episodes(str(tmp_path))  # no episodes/ subdir
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/shivamtyagi/PycharmProjects/nexus-memory
.venv/bin/pytest tests/test_ui_server.py -v 2>&1 | head -30
```

Expected: `ImportError` — `cannot import name '_read_episodes'`

- [ ] **Step 3: Add `import sqlite3` to `server.py`**

Open `smriti_memcore/ui/server.py`. The imports block starts at line 13. `sqlite3` is NOT currently imported. Add it:

```python
import sqlite3
```

alongside the other stdlib imports (`import json`, `import os`, etc.). Verify:

```bash
grep "^import sqlite3" smriti_memcore/ui/server.py
```

Expected: one match.

- [ ] **Step 4: Implement `_read_episodes` in `server.py`**

Add after the `_read_palace` function (around line 572, before `# ── HTTP Handler`):

```python
def _read_episodes(storage_path: str) -> list:
    """Read episodes.db and return a list of episode dicts, newest first.

    storage_path is pre-resolved by launch() — no expanduser() needed here.
    fetchall() materialises all rows into a Python list before conn.close(),
    so the for-loop runs safely after the finally block.
    """
    db_file = Path(storage_path) / "episodes" / "episodes.db"
    if not db_file.exists():
        return []

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row  # enables named column access: row["salience_json"] not row[N]
    try:
        rows = conn.execute(
            "SELECT id, content, timestamp, source, salience_json, consolidated "
            "FROM episodes ORDER BY timestamp DESC"
            # NOTE: projection order differs from DDL order (DDL has salience_json before source).
            # Always use named access (row["salience_json"], row["source"]) — never positional
            # indices — to avoid silent column swaps.
        ).fetchall()  # fetchall() returns a plain list — safe to use after conn.close()
    finally:
        conn.close()

    episodes = []
    for row in rows:
        try:
            # None or "{}" short-circuits before json.loads for NULL columns.
            # except TypeError is a defence against unexpected non-string types.
            salience = json.loads(row["salience_json"] or "{}").get("composite", 0.0)
        except (json.JSONDecodeError, TypeError):
            salience = 0.0
        episodes.append({
            "id":           row["id"],
            "content":      row["content"],
            "timestamp":    row["timestamp"],
            "source":       row["source"],
            "salience":     salience,
            "consolidated": bool(row["consolidated"]),  # SQLite stores 0/1 integers
        })
    return episodes
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_ui_server.py -v
```

Expected: all 6 tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add smriti_memcore/ui/server.py tests/test_ui_server.py
git commit -m "feat: add _read_episodes() — reads episodes.db for UI episode feed"
```

---

## Task 2: `/api/episodes` endpoint

**Files:**
- Modify: `smriti_memcore/ui/server.py` (`_Handler.do_GET`, around line 590)
- Modify: `tests/test_ui_server.py` (add 2 endpoint tests)

- [ ] **Step 1: Write failing endpoint tests**

Append to `tests/test_ui_server.py`:

```python
import threading
import urllib.request
from http.server import HTTPServer
from smriti_memcore.ui.server import _Handler


def _start_server(storage_path: str, port: int):
    """Start a test UI server in a daemon thread."""
    class Handler(_Handler):
        pass
    Handler.storage_path = storage_path
    server = HTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def test_api_episodes_endpoint_returns_json(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "test content", "2026-04-12T10:00:00", '{"composite":0.4}',
         "user_stated", None, 0, None, 1, None),
    ])
    server = _start_server(storage, 17799)
    try:
        with urllib.request.urlopen("http://127.0.0.1:17799/api/episodes") as resp:
            assert resp.status == 200
            assert "application/json" in resp.headers["Content-Type"]
            data = json.loads(resp.read())
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["id"] == "id-1"
            assert data[0]["consolidated"] is True
    finally:
        server.shutdown()


def test_api_episodes_empty_when_no_db(tmp_path):
    server = _start_server(str(tmp_path), 17800)
    try:
        with urllib.request.urlopen("http://127.0.0.1:17800/api/episodes") as resp:
            data = json.loads(resp.read())
            assert data == []
    finally:
        server.shutdown()
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
.venv/bin/pytest tests/test_ui_server.py::test_api_episodes_endpoint_returns_json tests/test_ui_server.py::test_api_episodes_empty_when_no_db -v
```

Expected: `FAIL` — 404 (route not yet added)

- [ ] **Step 3: Add `/api/episodes` route to `do_GET`**

In `_Handler.do_GET`, after the `/api/graph` block (around line 593):

```python
elif path == "/api/episodes":
    data = _read_episodes(self.storage_path)
    body = json.dumps(data).encode()
    self._respond(200, "application/json", body)
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
.venv/bin/pytest tests/test_ui_server.py -v
```

Expected: all 8 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/ui/server.py tests/test_ui_server.py
git commit -m "feat: add /api/episodes endpoint to UI server"
```

---

## Task 3: Episode Feed tab — HTML + CSS

**Files:**
- Modify: `smriti_memcore/ui/server.py` (embedded `_HTML` string)

- [ ] **Step 1: Add the tab button**

Find the tab bar (around line 160–164) in `_HTML`:

```html
<div class="tabs">
  <div class="tab active" id="tab-graph" onclick="showTab('graph')">🏛️ Semantic Palace</div>
  <div class="tab" id="tab-table" onclick="showTab('table')">📋 Memory Table</div>
  <div class="tab" id="tab-stats" onclick="showTab('stats')">📊 Statistics</div>
</div>
```

Add the 4th tab:

```html
<div class="tabs">
  <div class="tab active" id="tab-graph" onclick="showTab('graph')">🏛️ Semantic Palace</div>
  <div class="tab" id="tab-table" onclick="showTab('table')">📋 Memory Table</div>
  <div class="tab" id="tab-stats" onclick="showTab('stats')">📊 Statistics</div>
  <div class="tab" id="tab-episodes" onclick="showTab('episodes')">📼 Episode Feed</div>
</div>
```

- [ ] **Step 2: Add the view container**

The existing `view-table` CSS (in the `<style>` block) is:
```css
#view-table{flex:1;overflow:hidden;display:none;flex-direction:column;}
```

After the `<!-- STATS VIEW -->` block (around line 213–215), add a container styled consistently:

```html
  <!-- EPISODE FEED VIEW -->
  <div id="view-episodes" style="display:none;flex:1;overflow:hidden;flex-direction:column;">
    <div class="tbl-wrap">
      <table id="episodes-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Content</th>
            <th>Source</th>
            <th>Salience</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="episodes-body">
          <tr><td colspan="5" style="color:var(--muted);text-align:center">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>
```

Note: `flex:1;overflow:hidden;flex-direction:column` matches `#view-table` exactly so `showTab()` setting `display:'flex'` works correctly.

- [ ] **Step 3: Add `.status-pending` CSS**

Find `.status-ok` in the `<style>` block. Add `.status-pending` immediately after it:

```css
.status-pending{background:rgba(245,158,11,.15);color:var(--gold);padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600;}
```

`--gold` is `#f59e0b` — the amber design token already defined in `:root`.

- [ ] **Step 4: Smoke-test the tab renders**

```bash
.venv/bin/python3 -m smriti_memcore.ui --storage ~/.nexus/global --port 7799 --no-browser &
```

Open `http://127.0.0.1:7799`. Confirm:
- `📼 Episode Feed` tab appears in tab bar
- Clicking it shows the empty table with "Loading…" (no JS yet — this is expected)

Kill the server: `kill %1`

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/ui/server.py
git commit -m "feat: add Episode Feed tab HTML and CSS to UI"
```

---

## Task 4: Episode Feed tab — JavaScript

**Files:**
- Modify: `smriti_memcore/ui/server.py` (embedded `<script>` section)

- [ ] **Step 1: Update `showTab()` to include `'episodes'`**

Find `showTab` (around line 504):

```javascript
function showTab(tab){
  ['graph','table','stats'].forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active',t===tab);
    document.getElementById('view-'+t).style.display=t===tab?(t==='graph'?'flex':'flex'):'none';
  });
  if(tab==='graph'&&window._nodes){buildGraph(graphData);}
}
```

Replace with:

```javascript
function showTab(tab){
  ['graph','table','stats','episodes'].forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active',t===tab);
    document.getElementById('view-'+t).style.display=t===tab?'flex':'none';
  });
  if(tab==='graph'&&window._nodes){buildGraph(graphData);}
  if(tab==='episodes'&&!episodesLoaded){buildEpisodes();}
}
```

- [ ] **Step 2: Add `episodesLoaded` flag and `buildEpisodes()` function**

Add before the `// ── Boot` comment (around line 512):

```javascript
// ── Episode Feed ────────────────────────────────────────────
let episodesLoaded = false;

async function buildEpisodes(){
  try {
    const res = await fetch('/api/episodes');
    if(!res.ok) throw new Error('HTTP ' + res.status);
    const episodes = await res.json();
    episodesLoaded = true;
    const tbody = document.getElementById('episodes-body');
    if(!episodes.length){
      tbody.innerHTML='<tr><td colspan="5" style="color:var(--muted);text-align:center">No episodes found.</td></tr>';
      return;
    }
    tbody.innerHTML = episodes.map(ep=>`
      <tr>
        <td style="white-space:nowrap;font-family:var(--mono);font-size:11px">${ep.timestamp.replace('T',' ').slice(0,16)}</td>
        <td>${ep.content}</td>
        <td>${ep.source}</td>
        <td>${ep.salience.toFixed(3)}</td>
        <td><span class="${ep.consolidated?'status-ok':'status-pending'}">${ep.consolidated?'✓ consolidated':'⏳ pending'}</span></td>
      </tr>`).join('');
  } catch(err) {
    console.error('buildEpisodes failed:', err);
    document.getElementById('episodes-body').innerHTML=
      '<tr><td colspan="5" style="color:var(--red)">Error loading episodes.</td></tr>';
  }
}
```

- [ ] **Step 3: Reset `episodesLoaded` in `refreshData()`**

Find `refreshData()` (around line 236):

```javascript
async function refreshData(){
  const res = await fetch('/api/graph');
```

Add `episodesLoaded = false;` as the first line:

```javascript
async function refreshData(){
  episodesLoaded = false;
  const res = await fetch('/api/graph');
```

- [ ] **Step 4: Full smoke test**

```bash
.venv/bin/python3 -m smriti_memcore.ui --storage ~/.nexus/global --port 7799 --no-browser &
```

Open `http://127.0.0.1:7799`. Click `📼 Episode Feed`. Verify:

- Table populates with all episodes from `episodes.db` (expect 49 rows)
- Timestamps show as `YYYY-MM-DD HH:MM`
- Full content visible (no truncation)
- Source column shows `user_stated` etc.
- Salience shows 3 decimal places
- Status shows green `✓ consolidated` or amber `⏳ pending` badge
- Click `↻ Refresh` then switch back to Episodes — data reloads (network request visible in DevTools)
- Other tabs (Palace, Table, Stats) still work correctly

Kill the server: `kill %1`

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/ui/server.py
git commit -m "feat: Episode Feed JS — lazy load, buildEpisodes, showTab + refreshData update"
```

---

## Task 5: Version bump and PyPI release

**Files:**
- Modify: `pyproject.toml` only — `smriti_memcore/__init__.py` derives `__version__` from `importlib.metadata`, so `pyproject.toml` is the single source of truth. Do NOT search for a hardcoded `__version__` string.

- [ ] **Step 1: Bump version to 1.0.7**

In `pyproject.toml`, change:
```toml
version = "1.0.6"
```
to:
```toml
version = "1.0.7"
```

- [ ] **Step 2: Run full test suite**

```bash
.venv/bin/pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests pass including `tests/test_ui_server.py`

- [ ] **Step 3: Commit and push**

```bash
git add pyproject.toml
git commit -m "chore: bump to v1.0.7 — Episode Feed tab"
git push
```

- [ ] **Step 4: Build**

```bash
rm -rf dist/ && .venv/bin/python3 -m build 2>&1 | tail -3
```

Expected: `Successfully built smriti_memcore-1.0.7.tar.gz and smriti_memcore-1.0.7-py3-none-any.whl`

- [ ] **Step 5: Publish to PyPI**

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=<your-pypi-token> .venv/bin/python3 -m twine upload dist/smriti_memcore-1.0.7*
```

- [ ] **Step 6: Upgrade both venvs**

```bash
~/.nexus/venv/bin/pip3 install smriti-memcore==1.0.7 --no-cache-dir --quiet
.venv/bin/pip3 install smriti-memcore==1.0.7 --no-cache-dir --quiet
```

Verify:

```bash
~/.nexus/venv/bin/pip3 show smriti-memcore | grep Version
.venv/bin/pip3 show smriti-memcore | grep Version
```

Expected: `Version: 1.0.7` for both.
