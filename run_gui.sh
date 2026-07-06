#!/usr/bin/env bash
# Launch the interactive MuJoCo GUI with a hardware-accelerated X11 window
# (runs under XWayland on the AMD iGPU). Fixes the crippled pip "Wayland/OSMesa"
# GLFW by pointing at the system libglfw3 (X11 backend).
set -e

SYS_GLFW=/usr/lib/x86_64-linux-gnu/libglfw.so.3
VENV_PY=/home/balaji/venvs/mirte_rl/bin/python

if [ ! -e "$SYS_GLFW" ]; then
  echo "ERROR: system GLFW (X11) not found at $SYS_GLFW"
  echo "Install it once with:"
  echo "    sudo apt update && sudo apt install -y libglfw3 libglfw3-dev"
  exit 1
fi

# Force the X11-backend system GLFW instead of the pip Wayland/OSMesa build,
# and force GL onto the AMD Mesa (radeonsi) driver, not the absent NVIDIA one.
export PYGLFW_LIBRARY="$SYS_GLFW"
export MUJOCO_GL=glfw
export __GLX_VENDOR_LIBRARY_NAME=mesa
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

cd /home/balaji/mirte_balance_rl
exec "$VENV_PY" interactive_gui.py "$@"
