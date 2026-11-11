#!/bin/bash
# Publish a `calib` release: build the single-file calib from the current commit and attach it to a GitHub release of
# foundationbot/Calibration, where `calib --update` and new installs download it.
# Needs: gh (logged in, with push rights), the commit pushed, a clean tree. Usage: tools/calibration/release.sh [TAG]
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "$(git status --porcelain)" ]; then echo "release: the tree is not clean; commit first" >&2; exit 1; fi
sha=$(git rev-parse HEAD)
if ! git branch -r --contains "$sha" | grep -q .; then echo "release: $sha is not pushed yet" >&2; exit 1; fi
tag=${1:-v$(date +%Y.%m.%d)-$(git rev-parse --short=8 HEAD)}
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
python3 -m calibtool.cli --install --install-dir "$out" >/dev/null
version=$("$out/calib" --version)
echo "built: $version"
gh release create "$tag" "$out/calib" --repo foundationbot/Calibration --target "$sha" --title "calib $tag" \
  --notes "Single-file calib built from $sha ($version). Install or update: see README.md."
echo "released $tag"
