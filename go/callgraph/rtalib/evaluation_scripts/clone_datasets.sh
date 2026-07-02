#!/usr/bin/env bash
# Clone the OSS Go projects listed in targets.csv and pre-populate their
# module caches so the RTA sweep can run offline.
#
# Usage:
#   ./clone_datasets.sh [DATASETS_ROOT]
#
# DATASETS_ROOT defaults to ./datasets (relative to the script's dir).
#
# Idempotent: an existing dataset directory is left alone. Re-run after editing
# targets.csv to add new entries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The Uber monorepo's $PATH `go` is a wrapper that hardcodes GO111MODULE=off
# and sources a per-project env.sh, which makes `go mod download` against an
# OSS checkout silently no-op. Switch to the real go binary in GOROOT and
# force module mode.
_real_goroot="$(go env GOROOT 2>/dev/null || true)"
if [[ -x "${_real_goroot}/bin/go" ]]; then
  export PATH="${_real_goroot}/bin:${PATH}"
fi
export GO111MODULE=on
unset GOFLAGS GOPACKAGESDRIVER GOPACKAGESDRIVER_ULSP_MODE 2>/dev/null || true
TARGETS_FILE="${SCRIPT_DIR}/targets.csv"
DATASETS_ROOT="${1:-${SCRIPT_DIR}/datasets}"

if [[ ! -r "${TARGETS_FILE}" ]]; then
  echo "error: targets file not readable: ${TARGETS_FILE}" >&2
  exit 1
fi

mkdir -p "${DATASETS_ROOT}"
echo "datasets root: ${DATASETS_ROOT}"
echo

# Track which top-level repo dirs we've already cloned so two targets sharing
# a dir (kubernetes-kubelet + kubernetes-apiserver) only trigger one clone.
declare -A CLONED

clone_one() {
  local name="$1" repo="$2" tag="$3" dir="$4"
  # Top-level repo dir = first segment of dir (handles nested-module repos
  # like etcd/server which live inside the etcd checkout).
  local repo_dir="${dir%%/*}"
  local repo_path="${DATASETS_ROOT}/${repo_dir}"

  if [[ -n "${CLONED[${repo_dir}]:-}" ]]; then
    echo "[skip-clone] ${name} (repo ${repo_dir} already cloned for ${CLONED[${repo_dir}]})"
    return 0
  fi
  CLONED[${repo_dir}]="${name}"

  if [[ -d "${repo_path}/.git" ]]; then
    echo "[skip] ${repo_dir} (already exists at ${repo_path})"
    return 0
  fi

  echo "[clone] ${repo_dir} ${tag} <- ${repo}"
  git clone --depth 1 --branch "${tag}" "${repo}" "${repo_path}"
}

mod_download() {
  local name="$1" dir="$2"
  local mod_dir="${DATASETS_ROOT}/${dir}"
  if [[ ! -f "${mod_dir}/go.mod" ]]; then
    echo "[warn ] ${name}: no go.mod under ${mod_dir} (will likely fail at sweep time)"
    return 0
  fi
  echo "[mod  ] ${name}: go mod download in ${mod_dir}"
  (cd "${mod_dir}" && go mod download) || {
    echo "[warn ] ${name}: go mod download failed; sweep may need network access"
  }
}

# Read CSV (header + data rows), clone unique repos, then pre-download modules
# per target dir. CSV columns: name,repo,tag,dir,pkg (pkg unused here — the
# OSS sweep driver consumes it, not clone_datasets.sh).
declare -A MOD_DONE
{
  read -r _header  # discard the first line
  while IFS=, read -r name repo tag dir _pkg; do
    [[ -z "${name}" ]] && continue
    if [[ -z "${repo}" || -z "${tag}" || -z "${dir}" ]]; then
      echo "[warn ] malformed row, skipping: name=${name} repo=${repo} tag=${tag} dir=${dir}" >&2
      continue
    fi
    clone_one "${name}" "${repo}" "${tag}" "${dir}"
    if [[ -z "${MOD_DONE[${dir}]:-}" ]]; then
      MOD_DONE[${dir}]=1
      mod_download "${name}" "${dir}"
    fi
  done
} < "${TARGETS_FILE}"

echo
echo "done. datasets in: ${DATASETS_ROOT}"
ls -1 "${DATASETS_ROOT}"
