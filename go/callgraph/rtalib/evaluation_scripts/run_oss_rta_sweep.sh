#!/usr/bin/env bash
# Runs the oss_rta_bench binary across all targets in a manifest and writes
# one ${OUTPUT_DIR}/${target_name}.log per target. Structurally a fork of
# rta/run_rta_sweep.sh adapted for OSS dataset checkouts (no Bazel go_path,
# no Grail, no IDL discovery).
#
# Usage:
#   ./run_oss_rta_sweep.sh [options]
#
# Options:
#   -targets FILE   Path to targets manifest. Default: ./targets.csv
#                   CSV with header row; columns: name,repo,tag,dir,pkg
#   -datasets DIR   Root containing cloned OSS repos. Default: ./datasets
#                   (the same DATASETS_ROOT that clone_datasets.sh used)
#   -output DIR     Output directory for per-target logs.
#                   Default: /tmp/oss_rta_stats_$(date +%Y%m%d_%H%M%S)
#   -flavors LIST   Comma-separated RTA flavors.
#                   Default: srta,srta_opt,srta_struct,srta_kumo,srta_kumo_random,prta_kumo_nonblocking
#   -workers LIST   Comma-separated worker counts.
#                   Default: 1,2,4,8,16,32,64
#   -timeout SEC    Per-target timeout (passed to `timeout` cmd). Default: 5400
#   -binary PATH    Pre-built oss_rta_bench binary. If unset, the script tries:
#                     1. ./oss_rta_bench next to this script
#                     2. bazel-bin/.../oss_rta_bench
#                     3. `bazel build` of the target (cached after first run)
#   -force          Re-run targets even if their .log already shows a completion marker.
#   -dry-run        Print the commands that would be run, then exit.
#   -h | --help     Show this help.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The Uber monorepo's $PATH `go` is a wrapper that hardcodes GO111MODULE=off
# and sources a per-project env.sh, which breaks `go list` against OSS
# checkouts. Switch to the real go binary in GOROOT and force module mode for
# everything we and the harness binary's go/packages driver invoke.
_real_goroot="$(go env GOROOT 2>/dev/null || true)"
if [[ -x "${_real_goroot}/bin/go" ]]; then
  export PATH="${_real_goroot}/bin:${PATH}"
fi
export GO111MODULE=on
unset GOFLAGS GOPACKAGESDRIVER GOPACKAGESDRIVER_ULSP_MODE 2>/dev/null || true

TARGETS_FILE="${SCRIPT_DIR}/targets.csv"
DATASETS_DIR="${SCRIPT_DIR}/datasets"
OUTPUT_DIR=""
FLAVORS="srta,srta_opt,srta_struct,srta_kumo,srta_kumo_random,prta_kumo_nonblocking"
WORKERS="1,2,4,8,16,32,64"
TIMEOUT=5400
BINARY=""
FORCE=0
DRY_RUN=0

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -targets)  TARGETS_FILE="${2:?}"; shift 2;;
    -datasets) DATASETS_DIR="${2:?}"; shift 2;;
    -output)   OUTPUT_DIR="${2:?}";   shift 2;;
    -flavors)  FLAVORS="${2:?}";      shift 2;;
    -workers)  WORKERS="${2:?}";      shift 2;;
    -timeout)  TIMEOUT="${2:?}";      shift 2;;
    -binary)   BINARY="${2:?}";       shift 2;;
    -force)    FORCE=1;               shift;;
    -dry-run)  DRY_RUN=1;             shift;;
    -h|--help) usage 0;;
    *) echo "unknown arg: $1" >&2; usage 1;;
  esac
done

if [[ ! -r "${TARGETS_FILE}" ]]; then
  echo "error: targets file not readable: ${TARGETS_FILE}" >&2
  exit 1
fi

# Resolve absolute paths so the binary (which uses cfg.Dir for go list)
# doesn't care what the caller's cwd is.
DATASETS_DIR="$(cd "${DATASETS_DIR}" 2>/dev/null && pwd || echo "${DATASETS_DIR}")"

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="/tmp/oss_rta_stats_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUTPUT_DIR}"

# Binary resolution: -binary > script-local > go build.
GO_PKG="./cmd/oss_rta_bench"

resolve_binary() {
  if [[ -n "${BINARY}" ]]; then
    [[ -x "${BINARY}" ]] || { echo "error: -binary not executable: ${BINARY}" >&2; exit 1; }
    return
  fi
  if [[ -x "${SCRIPT_DIR}/oss_rta_bench" ]]; then
    BINARY="${SCRIPT_DIR}/oss_rta_bench"; return
  fi
  echo "[build] go build -o ${SCRIPT_DIR}/oss_rta_bench ${GO_PKG}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    (cd "${SCRIPT_DIR}" && go build -o oss_rta_bench "${GO_PKG}") >&2
  fi
  BINARY="${SCRIPT_DIR}/oss_rta_bench"
}
resolve_binary

# Read CSV manifest into parallel arrays (skip header row).
NAMES=(); REPOS=(); TAGS=(); DIRS=(); PKGS=()
{
  read -r _header  # discard the first line
  while IFS=, read -r name repo tag dir pkg; do
    [[ -z "${name}" ]] && continue
    if [[ -z "${repo}" || -z "${tag}" || -z "${dir}" || -z "${pkg}" ]]; then
      echo "[warn] malformed row, skipping: name=${name} repo=${repo} tag=${tag} dir=${dir} pkg=${pkg}" >&2
      continue
    fi
    NAMES+=("${name}"); REPOS+=("${repo}"); TAGS+=("${tag}"); DIRS+=("${dir}"); PKGS+=("${pkg}")
  done
} < "${TARGETS_FILE}"

TOTAL=${#NAMES[@]}
if [[ "${TOTAL}" -eq 0 ]]; then
  echo "error: no targets found in ${TARGETS_FILE}" >&2
  exit 1
fi

echo "sweep config:"
echo "  targets file  : ${TARGETS_FILE} (${TOTAL} targets)"
echo "  datasets dir  : ${DATASETS_DIR}"
echo "  output dir    : ${OUTPUT_DIR}"
echo "  flavors       : ${FLAVORS}"
echo "  workers       : ${WORKERS}"
echo "  timeout       : ${TIMEOUT}s"
echo "  binary        : ${BINARY}"
echo "  force re-run  : ${FORCE}"
echo

run_one() {
  local name="$1" dir="$2" pkg="$3"
  local log="${OUTPUT_DIR}/${name}.log"

  if [[ "${FORCE}" -eq 0 && -s "${log}" ]] && grep -q 'Finished CG construction-only evaluation' "${log}" 2>/dev/null; then
    echo "[skip] ${name} (already done: ${log})"
    return 200
  fi

  if [[ ! -d "${DATASETS_DIR}/${dir}" ]]; then
    echo "[skip-missing-dataset] ${name} (${DATASETS_DIR}/${dir} missing; run clone_datasets.sh)"
    return 201
  fi

  local cmd=(
    timeout "${TIMEOUT}"
    "${BINARY}"
    -dataset-root="${DATASETS_DIR}"
    -target-name="${name}"
    -target-dir="${dir}"
    -target-pkg="${pkg}"
    -flavors="${FLAVORS}"
    -workers="${WORKERS}"
  )

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] '; printf '%q ' "${cmd[@]}"; printf '> %q 2>&1\n' "${log}"
    return 0
  fi

  echo "[run]  ${name} -> ${log}"
  local start=$SECONDS
  if "${cmd[@]}" >"${log}" 2>&1; then
    local elapsed=$((SECONDS - start))
    echo "[ok]   ${name} (${elapsed}s)"
    return 0
  else
    local rc=$?
    local elapsed=$((SECONDS - start))
    if [[ $rc -eq 124 ]]; then
      echo "[TIMEOUT] ${name} (${elapsed}s, exceeded ${TIMEOUT}s)"
    else
      echo "[FAIL] ${name} (${elapsed}s, exit=$rc)"
    fi
    return $rc
  fi
}

OK=0; FAIL=0; SKIP=0; SKIP_MISSING=0; TIMED_OUT=0
for i in "${!NAMES[@]}"; do
  idx=$((i + 1))
  name="${NAMES[$i]}"
  dir="${DIRS[$i]}"
  pkg="${PKGS[$i]}"
  printf '\n=== [%d/%d] %s ===\n' "${idx}" "${TOTAL}" "${name}"
  set +e
  run_one "${name}" "${dir}" "${pkg}"
  rc=$?
  set -e
  case $rc in
    0)   OK=$((OK + 1));;
    124) TIMED_OUT=$((TIMED_OUT + 1));;
    200) SKIP=$((SKIP + 1));;
    201) SKIP_MISSING=$((SKIP_MISSING + 1));;
    *)   FAIL=$((FAIL + 1));;
  esac
done

echo
echo "=== sweep summary ==="
echo "  total              : ${TOTAL}"
echo "  ok                 : ${OK}"
echo "  skipped (done)     : ${SKIP}"
echo "  skipped (no data)  : ${SKIP_MISSING}"
echo "  failed             : ${FAIL}"
echo "  timed out          : ${TIMED_OUT}"
echo "  output             : ${OUTPUT_DIR}"
