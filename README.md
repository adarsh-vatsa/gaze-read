# gaze-read

Webcam gaze tracking for Claude Code. It watches which parts of Claude's reply you actually read and tells Claude in your next prompt.

```
Gaze tracker (webcam estimate): the user read about 35% of your previous reply.
Read: "The tracker counts how long you look at…", "Run the setup script once…"
Skimmed: "Press space and follow every dot…"
Skipped: "Limits: webcam gaze error is…", "Files are saved in…" (2 never scrolled into view)
Over the last 12 replies, the user read: first paragraph 80%, prose 40%, list items 20%, code blocks 10%.
This is about how you report to the user, not about the work: keep the work as thorough as before, and in the final message keep the kind of content the user reads and cut the kind they skip.
```

macOS only. Built-in webcam. No camera pictures are stored.

## Install

```
/plugin marketplace add adarsh-vatsa/gaze-read
/plugin install gaze-read@gaze-read
```

Then ask Claude to set up the gaze tracker. It runs these in order.

| Step | Command |
|---|---|
| Python env and face model | `scripts/setup.sh` |
| Camera and screen check | `scripts/gaze test` |
| Calibration (follow 21 dots, about 40 s) | `scripts/gaze calibrate` |
| Start tracking | `scripts/gaze start` |
| Stop tracking | `scripts/gaze stop` |
| Status | `scripts/gaze status` |
| Accuracy on your own clicks | `scripts/gaze eval` |

The app that runs the tracker needs **Camera** and **Screen & System Audio Recording** permission in System Settings → Privacy & Security. Inside a terminal app the commands run in place. From the Claude desktop app or another GUI host, the camera commands open in Terminal.app, which can ask macOS for access.

## How it works

```
webcam ──► MediaPipe face + iris ──► gaze point on screen ──┐
clicks, typing, dictation ──► on-the-fly drift correction ─┤
front window ──► screenshot ──► macOS Vision text lines ───┼──► UserPromptSubmit hook ──► report in your prompt
Claude's final message ──► paragraphs, list items, code ───┘
```

1. **Gaze.** MediaPipe Face Landmarker finds 478 face points including both irises. The iris position inside each eye plus head pose go through a per-user ridge regression fitted on calibration dots.
2. **Drift correction on the fly.** At every mouse click you look at the cursor. While you type or dictate (Superwhisper) into the prompt box you look at the box. Each such moment is a known gaze point. The tracker stores the error and your head pose at that moment and fits a correction that follows your posture. Time in a posture it has never seen counts as uncertain, never as skipped.
3. **Where the text is.** About once a second the tracker captures the front reply window and reads the text lines with macOS Vision when the picture changes. While you scroll it captures every 0.25 s and shifts the known lines by the measured image offset until a fresh read arrives.
4. **The report.** On each prompt the hook splits Claude's final message of the previous turn into blocks, matches screen lines to blocks by their words and gives each gaze sample to the nearest line. A block counts as read at half its normal reading time (230 words per minute), skimmed at 15 to 50 percent, skipped below that. Every report also goes into a local profile that gives the "over the last N replies" line.

The report changes how Claude writes its final message. The work itself stays as thorough as before.

## Results so far

All numbers come from one person on one MacBook Pro 16" (1728 × 1117 pt, 34.4 × 22.3 cm) during the first build session. Sample sizes are small. Treat them as a first look.

### Calibration and screen reading

| Check | Result |
|---|---|
| Face found by the camera | 15 of 15 frames |
| Calibration, 17 usable dots, leave-one-dot-out error | 196 pt |
| Calibration, error on 4 fresh dots | 71 pt (about 1.4 cm) |
| Text lines read per capture of the reply window | 35 to 42 |
| macOS Vision time for one capture | about 0.7 s |
| Paragraphs of a real reply found on screen | 3 of 3 |

### Drift is the main problem

Within about 20 minutes of calibrating, the raw estimate sat 1,000 to 1,300 pt off vertically. The head pose showed the cause. The head was pitched about 17° away from the calibration posture. Click points and typing points agreed on the size of the error, so typing in the prompt box is a usable known gaze point.

Early reports said 0 to 5 percent read for replies that had clearly been read. The causes were this drift and a bug that took Superwhisper's menu bar icon for its recording window. Once the click correction was running, one report listed exactly the paragraph and the step that the next prompt went on to answer.

### Held-out accuracy on real clicks

`gaze eval` trains on the earlier 70 percent of known gaze points and tests on the later 30 percent.

| Model | Known points | Median error on later points |
|---|---|---|
| Dot calibration + click shift | 16 | **58 pt** |
| Geometric model | 16 | 1,122 pt |
| Geometric model | 19 | 102 pt |
| Geometric model | 29 | 344 pt |

The simple model wins. A paragraph is about 60 to 150 pt tall, so 58 pt is enough to tell paragraphs apart.

### The geometric model is off by default

The geometric model treats gaze as a 3D ray from the eyes, using MediaPipe's head pose in centimetres, and intersects it with the screen plane. About 8 personal numbers are learned from clicks. On synthetic data it stayed within 14 to 40 pt across head tilts of 17° and distance changes of 15 cm after 15 training points. On real data it failed (table above). Likely causes are assumptions about MediaPipe's camera model (63° field of view, axis conventions) and too few points. It ships behind `"geo_model": true` in `~/.claude/gaze-read/config.json`, and even then it only switches on after it beats 100 pt on held-out points. Fixing it is the most useful open contribution.

### Synthetic tests

| Test | Result |
|---|---|
| Hook math, 36 s on a 460-word reply | 30% (expected 30%) |
| Per-block report on a fake reply with fake gaze | each block in the right group |
| Click shift, raw estimate 850 pt off | 0 pt after correction |
| Posture-aware correction, error up to 780 pt | 62 pt after 10 clicks, 37 pt after 25 |
| Scroll offset, text moved 60 px | measured 60 px |

## Known issues

- Only the main display is calibrated. Gaze on other monitors counts as off-screen.
- Dictation is detected by Superwhisper's recording window. Other dictation tools give no known gaze points yet.
- Hooks come from the Claude Code config directory that the host uses. A host that sets `CLAUDE_CONFIG_DIR` reads a different `settings.json`. The plugin install avoids this.
- Short neighbouring blocks such as one-line list items can swap.

## Files

| Path | Contents |
|---|---|
| `plugins/gaze-read/skills/gaze-read/SKILL.md` | the skill Claude loads |
| `plugins/gaze-read/skills/gaze-read/scripts/tracker.py` | calibration, tracker, drift correction, screen reading, geometric model |
| `plugins/gaze-read/skills/gaze-read/scripts/hook.py` | the UserPromptSubmit hook (standard library only) |
| `plugins/gaze-read/hooks/hooks.json` | registers the hook |
| `~/.claude/gaze-read/` | local data (config, calibration, last 30 min of gaze and screen text, profile) |

## Contributing

Issues and pull requests are welcome. Good places to start are the geometric model, other monitors, other dictation tools, and Linux or Windows support. Please include `gaze eval` output with any change to gaze estimation.

## License

MIT
