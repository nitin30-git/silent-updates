#!/bin/sh
# Usage: sh set_username.sh your-github-username
# Replaces the YOUR-USERNAME placeholder in the README, paper, citation file and tool docs.
[ -z "$1" ] && { echo "usage: sh set_username.sh your-github-username"; exit 1; }
for f in README.md CITATION.cff paper/silent_updates.tex tool/README.md tool/examples/model-drift.yml; do
  sed -i.bak "s/YOUR-USERNAME/$1/g" "$f" && rm -f "$f.bak"
done
grep -rl "YOUR-USERNAME" . --exclude=set_username.sh || echo "done: all placeholders replaced"
