#!/usr/bin/env bash
# fetch-gitleaks.sh — obtain the pinned gitleaks binary for a secret-scan job.
#
# CANONICAL COPY: qontinui-coord/.github/scripts/fetch-gitleaks.sh, tested by
# .github/scripts/test-fetch-gitleaks.sh (ci.yml, ci-scripts-tests). The same
# file is copied BYTE-IDENTICAL into every qontinui repo that runs gitleaks
# (plan 2026-11-09-gitleaks-release-cdn-504-reds-prs-and-blocks-the-merge-train).
# Edit it here first, then copy it; never edit a copy alone.
#
# WHY: the scan used to `curl … | tar …` the release tarball on every job. A
# GitHub release-CDN 504 (2026-11-08, 2026-11-09; coord dossier
# gitleaks-red-not-a-leak) then failed the required check with a bare
# `tar` exit 2 that read as a leak finding.
#
# SOURCES, in order. The cached tarball is copied into this job's private
# $RUNNER_TEMP and its sha256 checked THERE before use, so nothing can swap it
# between check and extract; the cache is not trusted and never required:
#   1. the runner's TOOL cache ${RUNNER_TOOL_CACHE:-$HOME/.cache}/qontinui-gitleaks/<sha256>.tar.gz
#      (the qontinui-coord workflow-lint.yml recipe for zizmor and actionlint):
#      a persistent self-hosted runner fetches each pinned version once;
#   2. the release CDN, with bounded retries. A GitHub-hosted runner starts
#      empty and always downloads; there is deliberately no actions/cache seed,
#      because a restored cache entry is a tar the required scan would have to
#      trust beyond the one checksummed file.
# A verified download fills the cache best-effort (mktemp in the cache
# directory, chmod 0644, then `mv -T`, an atomic rename), so a planted file, a
# directory at the key or an unwritable cache costs a download, never the scan.
#
# FAILURE: every failure here is infrastructure, never a finding. It prints one
# annotation whose TITLE carries "(not a verdict on this PR)" — the phrase
# qontinui-coord matches (pr_merge/author_feedback.rs
# NOT_A_VERDICT_ANNOTATION_MARKER, main_baseline_no_verdict.rs); keep it
# verbatim — and exits 75 (EX_TEMPFAIL), which no gitleaks outcome produces
# (the scan runs with --exit-code 3: leaks = 3; a gitleaks error is any other
# nonzero, 1 or 2 on a config panic).
# The workflow step that runs this must NOT be named with a prefix coord's
# INFRA_STEP_PREFIXES lists ("Set up ", "Setup ", "Cache", "Checkout", …): on
# merge-candidate CI that would make a failed fetch bypass-landable, and a
# secret scan that did not run must block.
#
# INPUT (env): GITLEAKS_VERSION, GITLEAKS_SHA256 (from the release's own
#   gitleaks_<version>_checksums.txt, never from a copy), RUNNER_TEMP,
#   GITHUB_OUTPUT; optional RUNNER_TOOL_CACHE.
# OUTPUT: `bin=<path to the gitleaks binary>` appended to $GITHUB_OUTPUT, only
#   after that binary has run `gitleaks version` successfully.
set -Eeuo pipefail

did_not_run() {
  echo "::error title=Gitleaks DID NOT RUN (not a verdict on this PR)::${1} THE SCAN DID NOT RUN. This is an infrastructure failure, not a leak finding; re-run the job."
  exit 75
}
trap 'did_not_run "gitleaks fetch failed at: ${BASH_COMMAND}."' ERR
tmp=""
trap 'rm -f -- "$tmp"' EXIT

[[ -n "${GITLEAKS_VERSION:-}" && -n "${GITLEAKS_SHA256:-}" ]] \
  || did_not_run "GITLEAKS_VERSION and GITLEAKS_SHA256 must both be set."
[[ "$GITLEAKS_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || did_not_run "GITLEAKS_SHA256 is not a sha256 (${GITLEAKS_SHA256})."
[[ -n "${RUNNER_TEMP:-}" && -n "${GITHUB_OUTPUT:-}" ]] \
  || did_not_run "RUNNER_TEMP and GITHUB_OUTPUT must both be set."

url="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
bin="${RUNNER_TEMP}/gitleaks-bin"
tgz="${bin}/gitleaks.tar.gz"
arch="$(uname -m)"
[[ "$arch" == "x86_64" ]] \
  || did_not_run "host arch is ${arch}, but the pinned gitleaks tarball is linux_x64 only."
cache="${RUNNER_TOOL_CACHE:-${HOME:-/nonexistent}/.cache}/qontinui-gitleaks"
hit="${cache}/${GITLEAKS_SHA256}.tar.gz"
mkdir -p "$bin"
rm -f -- "${bin}/gitleaks"

verify() { printf '%s  %s\n' "$GITLEAKS_SHA256" "$1" | sha256sum -c --status -; }

fill_cache() {
  # 0644: the cache may be shared by runner registrations running as other users.
  if mkdir -p "$cache" 2>/dev/null && tmp="$(mktemp "${cache}/.download.XXXXXX" 2>/dev/null)" \
      && cp -f -- "$tgz" "$tmp" && chmod 0644 "$tmp" && mv -fT -- "$tmp" "$hit" 2>/dev/null; then
    tmp=""
    echo "gitleaks: host cache filled (${hit})"
  else
    rm -f -- "$tmp"; tmp=""
    echo "::warning title=gitleaks cache not updated::${cache} is unusable, so every run on this runner downloads gitleaks; the scan is unaffected."
  fi
}

rm -f -- "$tgz"
cache_hit=0
if [[ -f "$hit" ]]; then
  if cp -f -- "$hit" "$tgz" 2>/dev/null && verify "$tgz"; then
    cache_hit=1
  else
    echo "::warning title=gitleaks cache entry unusable::${hit} is unreadable or fails its sha256; downloading instead."
  fi
fi

if [[ "$cache_hit" -eq 1 ]]; then
  echo "gitleaks ${GITLEAKS_VERSION}: host-cache hit (${hit}), sha256 verified, no download"
else
  rm -f -- "$tgz"
  rc=0
  curl --fail --silent --show-error --location \
    --connect-timeout 10 --max-time 60 \
    --retry 5 --retry-delay 5 --retry-max-time 120 --retry-all-errors \
    -o "$tgz" "$url" || rc=$?
  [[ "$rc" -eq 0 ]] || did_not_run "gitleaks ${GITLEAKS_VERSION} download failed (curl exit ${rc} from ${url})."
  verify "$tgz" || did_not_run "gitleaks ${GITLEAKS_VERSION} download does not match the pinned sha256 ${GITLEAKS_SHA256}."
  echo "gitleaks ${GITLEAKS_VERSION}: downloaded, sha256 verified"
  fill_cache
fi

tar -xzf "$tgz" -C "$bin" gitleaks 2>/dev/null \
  || did_not_run "the verified gitleaks ${GITLEAKS_VERSION} tarball did not extract a gitleaks binary."
[[ -x "${bin}/gitleaks" ]] || did_not_run "gitleaks ${GITLEAKS_VERSION} is not executable after extraction."
"${bin}/gitleaks" version >/dev/null 2>&1 \
  || did_not_run "the extracted gitleaks ${GITLEAKS_VERSION} does not run on this host."
echo "bin=${bin}/gitleaks" >> "$GITHUB_OUTPUT"
