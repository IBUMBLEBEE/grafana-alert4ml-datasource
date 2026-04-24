#!/usr/bin/env bash
# Prepare a release: bump version and regenerate CHANGELOG via git-cliff.
# Usage: ./scripts/release-pre.sh <X.Y.Z>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage: $0 <version>
Example: $0 0.2.8

Prepares a release by:
  1. Updating package.json and package-lock.json to <version>
  2. Regenerating CHANGELOG.md from git history via git-cliff
  3. Printing the commands to commit, tag, and push
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage; exit 0
fi

if [[ -z "${1:-}" ]]; then
  usage; exit 1
fi

VERSION="$1"

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: version must be X.Y.Z (e.g. 0.2.8)"
  exit 1
fi

# ── Prerequisite checks ──────────────────────────────────────
if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is required"
  exit 1
fi

if ! command -v git-cliff >/dev/null 2>&1; then
  echo "Error: git-cliff is not installed"
  echo "  cargo install git-cliff"
  echo "  or: npm install -g @git-cliff/git-cliff"
  exit 1
fi

cd "$REPO_ROOT"

# ── 1. Bump version in package.json + package-lock.json ──────
CURRENT="$(npm pkg get version | tr -d '"')"
if [[ "$CURRENT" == "$VERSION" ]]; then
  echo "package.json is already at $VERSION"
else
  echo "Bumping version: $CURRENT → $VERSION"
  npm version --no-git-tag-version "$VERSION" >/dev/null
  echo "✓ package.json and package-lock.json updated"
fi

# ── 2. Regenerate full CHANGELOG.md ──────────────────────────
echo "Generating CHANGELOG.md for tag v${VERSION}..."
git cliff --tag "v${VERSION}" --output CHANGELOG.md
echo "✓ CHANGELOG.md updated"

# ── 3. Print next steps ──────────────────────────────────────
echo
echo "Release v${VERSION} is ready. Next steps:"
echo
echo "  git diff                          # review"
echo "  git add package.json package-lock.json CHANGELOG.md"
echo "  git commit -m 'chore(release): v${VERSION}'"
echo "  git tag -a v${VERSION} -m 'Release v${VERSION}'"
echo "  git push origin HEAD"
echo "  git push origin v${VERSION}       # triggers CI release workflow"
