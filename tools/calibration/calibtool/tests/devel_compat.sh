#!/bin/bash
# devel-compatibility check for calibtool (plan section 12).
# usage: devel_compat.sh <devel tree> [pytest args...]
#   <devel tree>: a checkout of origin/devel (a worktree under ~/wt with GIT_LFS_SKIP_SMUDGE=1, never /tmp).
# Copies calibtool/ into the devel tree's tools/calibration/, reports which DEPS.txt modules devel lacks (those must be
# carried by the devel PR), copies them in too, runs calibtool's tests there, then removes everything it added, so the
# devel tree is left as it was.
set -u
DEV=${1:?usage: devel_compat.sh <devel tree> [pytest args...]}; shift
HERE=$(cd "$(dirname "$0")/.." && pwd)            # .../tools/calibration/calibtool
CAL=$(dirname "$HERE")                              # .../tools/calibration (this branch)
DCAL="$DEV/tools/calibration"
[ -d "$DCAL" ] || { echo "not a devel tree: $DCAL missing"; exit 2; }
added=()
cleanup() { for f in "${added[@]}"; do rm -rf "$DCAL/$f"; done; }
trap cleanup EXIT
[ -e "$DCAL/calibtool" ] && { echo "$DCAL/calibtool already exists; refusing to overwrite"; exit 2; }
cp -r "$HERE" "$DCAL/calibtool"; added+=(calibtool)
missing=0
while read -r dep; do
  dep=${dep%%#*}; dep=$(echo "$dep" | xargs); [ -z "$dep" ] && continue
  if [ ! -f "$DCAL/$dep" ]; then
    echo "devel lacks: $dep (carried by the devel PR)"; missing=$((missing + 1))
    mkdir -p "$DCAL/$(dirname "$dep")"; cp "$CAL/$dep" "$DCAL/$dep"; added+=("$dep")
  fi
done < "$HERE/DEPS.txt"
echo "devel lacks $missing of the DEPS.txt modules"
cd "$DEV" && python3 -m pytest tools/calibration/calibtool/tests -q -p no:cacheprovider "$@"
