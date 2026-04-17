#!/usr/bin/env bash
# Entrypoint for the qontinui eval-desktop container.
#
# Responsibilities:
#   1. Start Xvfb on :99
#   2. Wait until the X server is actually accepting connections (xdpyinfo poll)
#   3. Start a window manager (openbox; fluxbox as fallback if openbox fails)
#   4. Optionally start x11vnc (VNC_ENABLE=1) for interactive debugging
#   5. exec "$@" — the CMD or whatever the caller passed
#
# Env:
#   DISPLAY        - X display (default :99; inherited from Dockerfile ENV)
#   SCREEN_SIZE    - WxHxDepth for Xvfb (default 1920x1080x24 — fixed baseline
#                    matches proj_grounding_capture_pipeline.md baseline res)
#   VNC_ENABLE     - if "1", start x11vnc on 5900 (no password; local use only)
#   VNC_PASSWORD   - if set and VNC_ENABLE=1, x11vnc requires this password
#
# Exit code from exec'd child is propagated.

set -euo pipefail

: "${DISPLAY:=:99}"
: "${SCREEN_SIZE:=1920x1080x24}"
: "${VNC_ENABLE:=0}"

log() { printf '[entrypoint] %s\n' "$*" >&2; }

# 1. Xvfb ------------------------------------------------------------------
log "starting Xvfb on ${DISPLAY} (${SCREEN_SIZE})"
Xvfb "${DISPLAY}" -screen 0 "${SCREEN_SIZE}" -ac +extension RANDR -nolisten tcp &
XVFB_PID=$!

# 2. Wait for X server ----------------------------------------------------
# Poll xdpyinfo; give up after ~10s. If Xvfb crashed on startup (e.g. locked
# display file from a stale run), fail fast instead of hanging the container.
for i in $(seq 1 50); do
    if xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
        log "Xvfb is up (after ${i} polls)"
        break
    fi
    if ! kill -0 "${XVFB_PID}" 2>/dev/null; then
        log "Xvfb died during startup" >&2
        exit 1
    fi
    sleep 0.2
    if [ "${i}" = "50" ]; then
        log "Xvfb failed to come up within 10s" >&2
        exit 1
    fi
done

# 3. Window manager -------------------------------------------------------
# openbox-session doesn't exist in slim; openbox alone is fine. Fall back to
# fluxbox if openbox isn't installed or fails to start.
if command -v openbox >/dev/null 2>&1; then
    log "starting openbox"
    openbox &
elif command -v fluxbox >/dev/null 2>&1; then
    log "openbox missing; starting fluxbox"
    fluxbox &
else
    log "no window manager available — continuing without one" >&2
fi

# 4. VNC (optional) -------------------------------------------------------
if [ "${VNC_ENABLE}" = "1" ]; then
    if command -v x11vnc >/dev/null 2>&1; then
        VNC_ARGS=(-display "${DISPLAY}" -forever -shared -rfbport 5900)
        if [ -n "${VNC_PASSWORD:-}" ]; then
            VNC_PW_FILE="$(mktemp)"
            x11vnc -storepasswd "${VNC_PASSWORD}" "${VNC_PW_FILE}" >/dev/null
            VNC_ARGS+=(-rfbauth "${VNC_PW_FILE}")
            log "starting x11vnc on :5900 (password protected)"
        else
            VNC_ARGS+=(-nopw)
            log "starting x11vnc on :5900 (NO PASSWORD — local use only)"
        fi
        x11vnc "${VNC_ARGS[@]}" &
    else
        log "VNC_ENABLE=1 but x11vnc not installed" >&2
    fi
fi

# 5. Exec the command -----------------------------------------------------
if [ "$#" -eq 0 ]; then
    log "no command provided; defaulting to qontinui-hal-mcp http://0.0.0.0:7801"
    set -- qontinui-hal-mcp --http --host 0.0.0.0 --port 7801
fi

# If the command is hal-mcp and hal-mcp isn't installed, fail clearly rather
# than the cryptic "command not found" that a user would otherwise see.
if [ "${1}" = "qontinui-hal-mcp" ] && ! command -v qontinui-hal-mcp >/dev/null 2>&1; then
    log "qontinui-hal-mcp not installed in image — Phase 1 package missing from build context" >&2
    log "rebuild with qontinui-hal-mcp/ present in the build context" >&2
    exit 127
fi

log "exec: $*"
exec "$@"
