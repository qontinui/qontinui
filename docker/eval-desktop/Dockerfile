# Reproducible Linux desktop for grounding/WSM CI evaluation.
#
# WHY debian:stable-slim: smaller base (~80MB vs ubuntu:22.04 ~70MB after strip,
# but Debian's apt archive is leaner for our font+WM+X11 footprint and matches
# CI reproducibility better than a rolling Ubuntu LTS that can silently retag).
#
# Fixed X server + fixed fonts + fixed window manager = byte-identical renders
# across runs. That's the point — screenshots baked into grounding test data
# must look the same in CI as they did when captured.
#
# Ports:
#   7801 - qontinui-hal-mcp HTTP transport (Phase 1 deliverable)
#   5900 - x11vnc (only when VNC_ENABLE=1; local debugging only)

FROM debian:stable-slim

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- System packages ---
# Single apt-get layer, --no-install-recommends, lists cleaned at end.
# Grouped by purpose so future edits don't churn the layer unnecessarily.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # X server + X client utilities
    xvfb \
    x11-utils \
    x11-xserver-utils \
    # Window managers: openbox is primary, fluxbox as fallback
    openbox \
    fluxbox \
    # Input injection + screenshot CLIs (used by eval tooling / debugging)
    xdotool \
    scrot \
    # Optional VNC server for interactive debugging of failing eval runs
    x11vnc \
    # Python runtime (hal-mcp + eval scripts)
    python3 \
    python3-pip \
    python3-venv \
    # Fixed font set — reproducible rendering across runs
    fonts-dejavu \
    fonts-liberation \
    fonts-noto-core \
    # Runtime deps that apt pulls for the above (curl for healthchecks)
    curl \
    ca-certificates \
    procps \
 && rm -rf /var/lib/apt/lists/*

# --- Applications under test ---
# apps.list is a plain-text list of apt packages (one per line, # comments ok).
# Kept out-of-Dockerfile so CI can diff which apps changed between runs.
COPY apps.list /tmp/apps.list
RUN apt-get update && \
    grep -vE '^\s*(#|$)' /tmp/apps.list | xargs -r apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* /tmp/apps.list

# --- qontinui-hal-mcp install ---
# Phase 1 dep: the HAL-as-MCP package lives at qontinui-hal-mcp/ (sibling of
# qontinui-mcp/). Phase 1 must expose an HTTP transport — stdio won't work
# container-to-host. Until Phase 1 lands, this COPY + pip install -e will fail
# the build; that's intentional — we don't want to silently ship a broken image.
#
# Build context is expected to be the monorepo root (so the COPY can reach out
# of qontinui/docker/eval-desktop/). If run with the default `docker build
# ./qontinui/docker/eval-desktop` context, Phase 1 must be materialized into
# the build context another way (e.g. multi-stage, or context=repo-root).
#
# The COPY is guarded by || true so local builds can skip hal-mcp when Phase 1
# isn't on this branch yet. Production CI must build with --build-arg
# REQUIRE_HAL_MCP=1 to enforce presence.
ARG REQUIRE_HAL_MCP=0
COPY qontinui-hal-mcp /opt/qontinui-hal-mcp
RUN if [ -f /opt/qontinui-hal-mcp/pyproject.toml ]; then \
        pip install --break-system-packages -e /opt/qontinui-hal-mcp ; \
    elif [ "$REQUIRE_HAL_MCP" = "1" ]; then \
        echo "REQUIRE_HAL_MCP=1 but qontinui-hal-mcp not present in build context" >&2; exit 1; \
    else \
        echo "qontinui-hal-mcp not present; image will lack hal-mcp server (Phase 1 pending)" >&2; \
    fi

# --- Entrypoint ---
COPY qontinui/docker/eval-desktop/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 7801 5900

# Default command: start the hal-mcp HTTP server on 7801. If hal-mcp is absent,
# the entrypoint exits non-zero with a clear message.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["qontinui-hal-mcp", "--http", "--host", "0.0.0.0", "--port", "7801"]
