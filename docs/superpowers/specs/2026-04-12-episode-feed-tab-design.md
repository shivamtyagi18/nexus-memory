# Episode Feed Tab — Design Spec

**Date:** 2026-04-12
**Status:** Approved

## Problem

The Smriti Memory Browser UI currently shows only palace memories (`palace.json` — 31 long-term memories). Raw episodic memories stored in `episodes.db` (49 rows) are invisible, making it impossible to see the full encoding history or debug why certain memories were or weren't promoted to the palace.

## Goal

Add a `📼 Episode Feed` tab to the UI that shows the raw episode buffer as a reverse-chronological table, with consolidation status clearly visible.

## Scope

- New `/api/episodes` server endpoint
- New `_read_episodes()` data function
- New `📼 Episode Feed` tab in the browser UI
- Update `showTab()` to include the new `'episodes'` tab ID

No changes to existing endpoints or existing tab behaviour.

Out of scope: search/filter, trajectory grouping, reflections display.

## Data Layer

### Storage location
```
<SMRITI_STORAGE_PATH>/episodes/episodes.db
```
Matches `EpisodeBuffer.__init__`: `os.path.join(storage_path, "episodes", "episodes.db")`.

### New function: `_read_episodes(storage_path: str) -> list[dict]`

Reads `episodes.db` using stdlib `sqlite3`. `storage_path` arrives pre-resolved (no `expanduser()` needed — `launch()` resolves it before storing on the handler class, matching the `_read_palace` convention).

Returns a list of dicts ordered by `timestamp DESC`:

```python
{
    "id":           str,    # UUID
    "content":      str,    # raw episode text
    "timestamp":    str,    # ISO8601 string
    "source":       str,    # "user_stated" | "inferred" | "direct" | "external"
    "salience":     float,  # composite extracted from salience_json
    "consolidated": bool,   # bool(row["consolidated"]) — SQLite stores 0/1 integers
}
```

**`salience_json` handling:** The column has no DEFAULT and may be `NULL`. Parse with:
```python
try:
    salience = json.loads(row["salience_json"] or "{}").get("composite", 0.0)
except (json.JSONDecodeError, TypeError):
    salience = 0.0
```
This guards against both `NULL` (`TypeError` from `json.loads(None)`) and malformed JSON.

**`consolidated` handling:** SQLite returns `0` or `1`. Always cast: `bool(row["consolidated"])`.

Returns `[]` if `episodes.db` does not exist (no exception raised).

### New endpoint: `GET /api/episodes`

Calls `_read_episodes(self.storage_path)` and returns `json.dumps(episodes_list)` with `Content-Type: application/json`. Added to `_Handler.do_GET` alongside the existing `/api/graph` and `/api/health` routes.

## UI Layer

### Tab bar

Add a 4th tab after Statistics:

```html
<div class="tab" id="tab-episodes" onclick="showTab('episodes')">📼 Episode Feed</div>
```

### View container

```html
<div id="view-episodes" style="display:none; flex:1; overflow:auto; padding:24px;">
  <table id="episodes-table">...</table>
</div>
```
Initial CSS must be `display:none` to match the existing show/hide convention used by all other view containers (`showTab()` drives visibility).

### Table columns

| Column | Field | Notes |
|--------|-------|-------|
| Timestamp | `timestamp` | Formatted as `YYYY-MM-DD HH:MM`, newest first |
| Content | `content` | Full content, no truncation |
| Source | `source` | Plain text |
| Salience | `salience` | 3 decimal places |
| Status | `consolidated` | Green `✓ consolidated` or amber `⏳ pending` badge |

Empty state: if `episodes` array is empty, show `<p>No episodes found.</p>`.

### Lazy loading

`buildEpisodes()` fetches `/api/episodes` only on first tab activation, guarded by a `let episodesLoaded = false` flag. The global header `↻ Refresh` button calls `refreshData()` which resets `episodesLoaded = false` in addition to re-fetching `/api/graph` — so the episode cache is cleared by the same global refresh, keeping the UX consistent with the other tabs. No separate per-tab refresh button.

### `showTab()` update

Extend the existing tab/view array from `['graph','table','stats']` to `['graph','table','stats','episodes']`.

### Styling

Reuses existing `.tbl-wrap`, `table`, `th`, `td`, `.status-ok` CSS classes for visual consistency with the Memory Table tab. The `⏳ pending` badge uses a new `.status-pending` class styled amber, consistent with existing badge patterns.

## Error handling

| Scenario | Behaviour |
|----------|-----------|
| `episodes.db` missing | `_read_episodes` returns `[]`; UI shows "No episodes found." |
| `salience_json` is NULL | Defaults to `0.0` (no crash) |
| `salience_json` malformed JSON | Defaults to `0.0` (no crash) |
| `consolidated` is 0/1 integer | Cast with `bool()` before serialising |
| JS fetch error | Logs to console; shows inline error message in the tab |

## Changes summary

| File | Change |
|------|--------|
| `smriti_memcore/ui/server.py` | Add `_read_episodes()` function |
| `smriti_memcore/ui/server.py` | Add `/api/episodes` route in `do_GET` |
| `smriti_memcore/ui/server.py` | Add `📼 Episode Feed` tab HTML and `#view-episodes` container |
| `smriti_memcore/ui/server.py` | Add `buildEpisodes()` JS function and `episodesLoaded` flag |
| `smriti_memcore/ui/server.py` | Update `showTab()` to include `'episodes'` |
| `smriti_memcore/ui/server.py` | Update `refreshData()` to reset `episodesLoaded = false` |

## Non-goals

- Trajectory grouping (future)
- Reflections display (future)
- Search/filter on episodes (future)
- Linking episodes to their corresponding palace memory (future)
- Per-tab refresh button (global refresh is sufficient)
