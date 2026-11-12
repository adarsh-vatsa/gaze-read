#!/usr/bin/env python3
"""UserPromptSubmit hook for gaze-read. Standard library only, silent when the tracker is off.

On each prompt it takes Claude's previous reply, splits it into blocks (paragraphs, list items,
code blocks, headings, tables), and works out how long the user's gaze sat on each block:

  gaze samples  (samples.csv: time, x, y, on-reply-area)       from the tracker
  screen text   (screen.jsonl: where each text line sits)       from the tracker
  reply blocks  (from the transcript)

Each screen text line is matched to a reply block by its words. Each gaze sample goes to the
nearest matched line. A block counts as read when the looking time reaches half its normal
reading time. A per-user profile (profile.jsonl) keeps these results across replies.

  hook.py prompt     the hook
  hook.py report     print the last report again (for checking)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

HOME = Path(os.environ.get("GAZE_READ_HOME", Path.home() / ".claude" / "gaze-read"))
STATE = HOME / "state"
SAMPLES = HOME / "samples.csv"
SCREEN = HOME / "screen.jsonl"
PROFILE = HOME / "profile.jsonl"
PIDFILE = HOME / "tracker.pid"
CONFIG = HOME / "config.json"
HOOKLOG = HOME / "hook.log"
LAST = HOME / "last_report.txt"

LINE_MATCH = 0.75     # share of a screen line's words that must appear in a block
GAZE_TO_LINE_PT = 45  # max vertical distance from gaze point to a text line
MAX_GAP_S = 0.5       # a sample counts for at most this long
READ, SKIM = 0.5, 0.15


def tracker_running():
    try:
        os.kill(int(PIDFILE.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def config():
    cfg = {"wpm": 230, "keep_s": 1800}
    try:
        cfg.update(json.loads(CONFIG.read_text()))
    except (OSError, ValueError):
        pass
    return cfg


def log(msg):
    try:
        with open(HOOKLOG, "a") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------- reply text

def is_real_prompt(entry):
    if entry.get("isMeta"):
        return False
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        kinds = {b.get("type") for b in content if isinstance(b, dict)}
        return "text" in kinds and "tool_result" not in kinds
    return False


def last_reply_text(transcript_path):
    """Text of Claude's final message in the most recent turn (not the progress lines between tool calls).
    One message can be logged as several entries that share message.id; those are joined."""
    try:
        lines = Path(transcript_path).read_text().splitlines()
    except OSError:
        return ""
    parts, msg_id = [], None
    for line in reversed(lines):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "user" and is_real_prompt(e) and msg_id is not None:
            break
        if e.get("type") != "assistant":
            continue
        m = e.get("message", {})
        texts = [b.get("text", "") for b in m.get("content", []) or []
                 if isinstance(b, dict) and b.get("type") == "text"]
        if not texts:
            if msg_id is not None and m.get("id") != msg_id:
                break
            continue
        if msg_id is None:
            msg_id = m.get("id") or "?"
        elif m.get("id") != msg_id:
            break
        parts.append("\n\n".join(texts))
    return "\n\n".join(reversed(parts))


LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


def split_blocks(text):
    """Markdown -> [(kind, text)], kind in heading/prose/list/code/table."""
    blocks, cur, kind, in_code = [], [], None, False

    def flush():
        nonlocal cur, kind
        if cur and "".join(cur).strip():
            blocks.append((kind or "prose", "\n".join(cur)))
        cur, kind = [], None

    for line in text.splitlines():
        if line.strip().startswith("```"):
            if in_code:
                cur.append(line)
                flush()
                in_code = False
            else:
                flush()
                cur, kind, in_code = [line], "code", True
            continue
        if in_code:
            cur.append(line)
            continue
        if not line.strip():
            flush()
            continue
        if line.lstrip().startswith("#"):
            flush()
            blocks.append(("heading", line))
            continue
        if line.lstrip().startswith("|"):
            if kind != "table":
                flush()
                kind = "table"
            cur.append(line)
            continue
        if LIST_RE.match(line) and not (line.startswith("  ") and kind == "list"):
            flush()
            kind = "list"
        elif kind is None:
            kind = "prose"
        cur.append(line)
    flush()
    return blocks


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def snippet(text, n=8):
    words = re.sub(r"[`*_#>|]", "", text).split()
    words = [w for w in words if not LIST_RE.match(w + " ")] or words
    s = " ".join(words[:n])
    return s + ("…" if len(words) > n else "")


# ---------------------------------------------------------------- gaze data

def read_tail(path, nbytes=8_000_000):
    if not path.exists():
        return []
    size = path.stat().st_size
    with open(path, "rb") as fh:
        fh.seek(max(0, size - nbytes))
        rows = fh.read().decode(errors="ignore").splitlines()
    return rows[1:] if size > nbytes else rows


def load_samples(start, end):
    out = []
    for r in read_tail(SAMPLES):
        p = r.split(",")
        if len(p) not in (4, 5):
            continue
        try:
            t = float(p[0])
        except ValueError:
            continue
        if start <= t <= end:
            gx = float(p[1]) if p[1] else None
            gy = float(p[2]) if p[2] else None
            known = len(p) == 4 or p[4] == "1"
            out.append((t, gx, gy, p[3] == "1", known))
    return out


def load_snapshots(start, end):
    snaps = []
    for r in read_tail(SCREEN):
        try:
            s = json.loads(r)
        except ValueError:
            continue
        if s["t"] <= end:
            snaps.append(s)
    # keep the one active at `start` plus all later ones
    before = [s for s in snaps if s["t"] <= start]
    return before[-1:] + [s for s in snaps if s["t"] > start]


def match_lines(snap, block_tokens):
    """[(cy, x0, x1, block_index)] for screen lines that belong to the reply."""
    out = []
    for text, x, y, w, h in snap["lines"]:
        lt = tokens(text)
        if len(lt) < 3:
            continue
        best, best_i = 0.0, None
        for i, bt in enumerate(block_tokens):
            score = sum(1 for t in lt if t in bt) / len(lt)
            if score > best:
                best, best_i = score, i
        if best >= LINE_MATCH:
            out.append((y + h / 2.0, x, x + w, best_i))
    return out


def analyse(blocks, start, end, wpm):
    """Per block: seconds of gaze, seconds needed, whether it was ever on screen."""
    block_tokens = [set(tokens(t)) for _, t in blocks]
    need = [max(1.0, len(t.split()) / wpm * 60.0) for _, t in blocks]
    dwell = [0.0] * len(blocks)
    on_screen = [False] * len(blocks)
    samples = load_samples(start, end)
    snaps = load_snapshots(start, end)
    if not samples:
        return None
    matched = [(s["t"], match_lines(s, block_tokens)) for s in snaps]
    for _, lines in matched:
        for *_, b in lines:
            on_screen[b] = True
    si = 0
    window_s = 0.0
    nxt_t = [s[0] for s in samples[1:]] + [end]
    unsure_s = 0.0
    for (t, gx, gy, inside, known), t2 in zip(samples, nxt_t):
        dt = min(max(t2 - t, 0.0), MAX_GAP_S)
        if gx is not None and not known:
            unsure_s += dt  # posture outside what the tracker knows: neither read nor skipped
            continue
        if not inside or gx is None:
            continue
        window_s += dt
        while si + 1 < len(matched) and matched[si + 1][0] <= t:
            si += 1
        if not matched or matched[si][0] > t:
            continue
        best = None
        for cy, x0, x1, b in matched[si][1]:
            d = abs(gy - cy)
            if d <= GAZE_TO_LINE_PT and x0 - 150 <= gx <= x1 + 150 and (best is None or d < best[0]):
                best = (d, b)
        if best:
            dwell[best[1]] += dt
    return {"dwell": dwell, "need": need, "on_screen": on_screen,
            "has_screen": any(on_screen), "window_s": window_s, "unsure_s": unsure_s,
            "span_s": max(1.0, end - start)}


# ---------------------------------------------------------------- profile

def position(i, n):
    return "first" if i == 0 else ("last" if i == n - 1 else "middle")


def save_profile(blocks, res):
    n = len(blocks)
    rec = {"t": round(time.time()), "blocks": [
        {"kind": k, "pos": position(i, n), "need": round(res["need"][i], 1),
         "frac": round(min(1.0, res["dwell"][i] / res["need"][i]), 2), "snip": snippet(t, 6)}
        for i, (k, t) in enumerate(blocks)]}
    with open(PROFILE, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def profile_summary(max_replies=30):
    if not PROFILE.exists():
        return None
    recs = [json.loads(l) for l in PROFILE.read_text().splitlines()[-max_replies:] if l.strip()]
    if len(recs) < 3:
        return None
    groups = {}
    for r in recs:
        for b in r["blocks"]:
            for key in (f"{b['pos']} paragraph" if b["pos"] != "middle" else None,
                        {"prose": "prose", "list": "list items", "code": "code blocks",
                         "table": "tables", "heading": None}.get(b["kind"])):
                if key:
                    g = groups.setdefault(key, [0.0, 0.0, 0])
                    g[0] += b["frac"] * b["need"]
                    g[1] += b["need"]
                    g[2] += 1
    order = ["first paragraph", "prose", "list items", "tables", "code blocks", "last paragraph"]
    parts = [f"{k} {round(100 * groups[k][0] / groups[k][1])}%" for k in order
             if k in groups and groups[k][2] >= 5 and groups[k][1] > 0]
    if not parts:
        return None
    return f"Over the last {len(recs)} replies, the user read: " + ", ".join(parts) + "."


# ---------------------------------------------------------------- report

def report(blocks, res):
    dwell, need = res["dwell"], res["need"]
    total_need = sum(need)
    pct = round(100 * sum(min(d, n) for d, n in zip(dwell, need)) / total_need / 5) * 5
    frac = [min(1.0, d / n) for d, n in zip(dwell, need)]
    read = [i for i, f in enumerate(frac) if f >= READ and blocks[i][0] != "heading"]
    skim = [i for i, f in enumerate(frac) if SKIM <= f < READ and blocks[i][0] != "heading"]
    skip = [i for i, f in enumerate(frac) if f < SKIM and blocks[i][0] != "heading"]
    unseen = sum(1 for i in skip if not res["on_screen"][i])

    def fmt(idx, limit=6):
        s = ", ".join(f'"{snippet(blocks[i][1])}"' for i in idx[:limit])
        return s + (f" and {len(idx) - limit} more" if len(idx) > limit else "")

    out = [f"Gaze tracker (webcam estimate): the user read about {pct}% of your previous reply."]
    if read:
        out.append("Read: " + fmt(read))
    if skim:
        out.append("Skimmed: " + fmt(skim))
    if skip:
        out.append("Skipped: " + fmt(skip) + (f" ({unseen} never scrolled into view)" if unseen else ""))
    if res["unsure_s"] > 0.2 * min(res["span_s"], 600):
        out.append(f"(Gaze was uncertain for {round(res['unsure_s'])} s because the user sat in an unfamiliar "
                   f"position; the read share may be higher.)")
    summ = profile_summary()
    if summ:
        out.append(summ)
    out.append("This is about how you report to the user, not about the work: keep the work as thorough as before, "
               "and in the final message keep the kind of content the user reads and cut the kind they skip.")
    return "\n".join(out)


def fallback_report(words, res):
    need = words / config()["wpm"] * 60.0
    pct = round(100 * min(1.0, res["window_s"] / need) / 5) * 5
    return f"Gaze tracker (webcam estimate): the user read about {pct}% of your previous reply."


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "report":
        print(LAST.read_text() if LAST.exists() else "No report yet.")
        return
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = {}
    if mode != "prompt":
        return  # the old "stop" mode is no longer used
    if not tracker_running():
        return
    cfg = config()
    sid = "".join(c for c in str(data.get("session_id", "default")) if c.isalnum() or c in "-_") or "default"
    STATE.mkdir(parents=True, exist_ok=True)
    sf = STATE / f"{sid}.json"
    now = time.time()
    try:
        prev = json.loads(sf.read_text()).get("prompt")
    except (OSError, ValueError):
        prev = None
    sf.write_text(json.dumps({"prompt": now}))
    start = max(prev or now - 600, now - float(cfg["keep_s"]))

    text = last_reply_text(data.get("transcript_path", ""))
    blocks = split_blocks(text)
    words = sum(len(t.split()) for _, t in blocks)
    if words < 20:
        log(f"prompt sid={sid} words={words}: too short, no report")
        return
    res = analyse(blocks, start, now, float(cfg["wpm"]))
    if res is None:
        log(f"prompt sid={sid}: no gaze samples")
        return
    if res["has_screen"]:
        msg = report(blocks, res)
        save_profile(blocks, res)
    else:
        msg = fallback_report(words, res)
    LAST.write_text(msg + "\n")
    log(f"prompt sid={sid} blocks={len(blocks)} screen={res['has_screen']} | " + msg.splitlines()[0])
    print(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"error: {e!r}")  # a hook must never block the user's prompt
