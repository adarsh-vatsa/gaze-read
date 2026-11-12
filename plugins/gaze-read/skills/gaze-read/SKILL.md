---
name: gaze-read
description: Webcam gaze tracker that measures which parts of Claude's last reply the user actually read, and adds a short reading report to their next prompt. Use when the user asks to set up, calibrate, start, stop, test, check or evaluate the gaze tracker, or asks about the reading report.
---

# gaze-read

A background webcam process estimates where the user looks on screen. The plugin's
**UserPromptSubmit** hook turns that into a reading report for Claude's previous reply.

## How it works

The tracker (`scripts/tracker.py run`, started with `gaze start`):

- reads the webcam about 10 times per second. MediaPipe finds the face and irises; a per-user
  mapping turns iris position and head pose into a screen point (`samples.csv`);
- corrects drift on the fly. At each mouse click the user looks at the cursor; while typing or
  dictating (Superwhisper) into the prompt box they look at the box. These known points, with
  the head pose at that moment, train a correction that follows the user's posture
  (`anchors.jsonl`). Posture far from anything seen is marked *uncertain*;
- about once per second captures the front reply window (a terminal, the Claude app, an editor),
  and when it changes reads the text lines with macOS Vision (`screen.jsonl`). While the user
  scrolls it captures every 0.25 s and moves the known lines by the measured image shift.

The hook (`scripts/hook.py prompt`) splits Claude's final message of the previous turn into
blocks (paragraphs, list items, code blocks, tables), matches screen lines to blocks by their
words, and gives each gaze sample to the nearest line. A block is *read* when the looking time
reaches half its normal reading time (230 wpm), *skimmed* at 15–50%, *skipped* below that.
The hook adds lines like these to the prompt:

```
Gaze tracker (webcam estimate): the user read about 35% of your previous reply.
Read: "…", "…"
Skimmed: "…"
Skipped: "…", "…" (2 never scrolled into view)
Over the last 12 replies, the user read: first paragraph 80%, prose 40%, list items 20%, …
This is about how you report to the user, not about the work: keep the work as thorough as before, and in the final message keep the kind of content the user reads and cut the kind they skip.
```

The hook prints nothing when the tracker is off.

When these lines appear in a prompt, they describe how the user reads the final message.
Change only how you report: keep the work itself (checks, tests, research) as thorough as before.

## Commands

The scripts live in this skill's `scripts/` folder. Run them with Bash, using the skill's base
directory: `G=<skill dir>/scripts/gaze`.

| Ask | Run |
|---|---|
| first-time setup (Python env + face model, about 1 min) | `<skill dir>/scripts/setup.sh` |
| check the camera and screen reading | `$G test` |
| calibrate (full-screen dots, about 40 s) | `$G calibrate` |
| turn tracking on / off | `$G start` / `$G stop` |
| status | `$G status` |
| accuracy on real data (train on earlier known points, test on later ones) | `$G eval` |
| show the last report | `/usr/bin/python3 <skill dir>/scripts/hook.py report` |

- Order for a new user: `setup.sh`, `test`, `calibrate`, `start`.
- `calibrate` opens a full-screen window. Tell the user to press SPACE and follow each dot with
  their eyes.
- macOS gives camera and screen access per app. In a terminal app the commands run in place, and
  macOS asks that terminal for access. From a desktop app's shell (the Claude app, T3 Code) the
  camera commands open in Terminal.app instead, and their output is copied to
  `~/.claude/gaze-read/last_<cmd>.txt`; read it from there when the command ends.
- The app that runs the tracker needs **Camera** and **Screen & System Audio Recording**
  permission (System Settings → Privacy & Security). Without screen recording, the report has
  only the percent line.

## Settings

`~/.claude/gaze-read/config.json` (created on first run):

| key | meaning |
|---|---|
| `camera` | OpenCV camera index; try 1 if 0 is an iPhone Continuity Camera |
| `apps` | window owner names (prefix match) where replies are read |
| `input_strip_pt` | bottom strip of the window (the prompt box) that does not count as reading |
| `wpm` | reading speed used for "normal reading time" |
| `geo_model` | experimental geometric model, off by default (see README) |

## Limits

- macOS only. Only the main display is calibrated.
- Webcam gaze error is roughly 1.5–4 cm, so the tracker knows blocks, not words.
- No camera pictures are saved. Gaze samples and screen text keep the last 30 minutes; known
  gaze points keep 4 hours; the latest window capture is overwritten each second.
