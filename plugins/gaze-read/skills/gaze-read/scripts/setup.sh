#!/bin/sh
# One-time setup: Python env with MediaPipe, OpenCV, PyObjC (Quartz, Vision) and SciPy, plus the
# MediaPipe face landmark model. Everything goes into ~/.claude/gaze-read (or $GAZE_READ_HOME).
set -e
[ "$(uname)" = "Darwin" ] || { echo "gaze-read needs macOS."; exit 1; }
D="${GAZE_READ_HOME:-$HOME/.claude/gaze-read}"
PKGS="mediapipe numpy scipy pyobjc-framework-Quartz pyobjc-framework-Vision"
mkdir -p "$D"
if command -v uv >/dev/null 2>&1; then
  [ -x "$D/venv/bin/python" ] || uv venv -q --python 3.11 "$D/venv"
  uv pip install -q --python "$D/venv/bin/python" $PKGS
else
  [ -x "$D/venv/bin/python" ] || python3 -m venv "$D/venv"
  "$D/venv/bin/python" -m pip install -q --upgrade pip
  "$D/venv/bin/python" -m pip install -q $PKGS
fi
[ -f "$D/face_landmarker.task" ] || curl -sSfL -o "$D/face_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
echo "Setup done in $D"
