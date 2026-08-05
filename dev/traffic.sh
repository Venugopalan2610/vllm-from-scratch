#!/usr/bin/env bash
# Current traffic (GitHub's rolling 14-day window) plus permanent counters.
# Accumulated history, if the workflow is running, lives on the `traffic` branch.
R="${1:-Venugopalan2610/vllm-from-scratch}"

echo "== $R =="
gh api "repos/$R" --jq '"stars: \(.stargazers_count)   forks: \(.forks_count)   watchers: \(.subscribers_count)"'

echo
echo "last 14 days (all GitHub keeps):"
gh api "repos/$R/traffic/clones" --jq '"  clones: \(.count) total, \(.uniques) unique"'
gh api "repos/$R/traffic/views"  --jq '"  views:  \(.count) total, \(.uniques) unique"'

echo
echo "top referrers:"
gh api "repos/$R/traffic/popular/referrers" \
  --jq '.[] | "  \(.referrer): \(.count) (\(.uniques) unique)"' 2>/dev/null | head -5

echo
echo "forks (permanent -- the metric that does not expire):"
gh api "repos/$R/forks" --jq '.[] | "  \(.owner.login)  \(.created_at[:10])"' 2>/dev/null | head -20

echo
echo "accumulated history (traffic branch):"
if git fetch -q origin traffic --depth=1 2>/dev/null && \
   git show origin/traffic:history.json > /tmp/_hist.json 2>/dev/null; then
  python3 -c "
import json
s = json.load(open('/tmp/_hist.json'))['summary']
for k in ('clones', 'views'):
    d = s[k]
    print(f\"  {k}: {d['total']} total, {d['total_uniques']} unique, over {d['days_recorded']} days\")"
else
  echo "  none yet -- the workflow needs the TRAFFIC_TOKEN secret."
  echo "  see .github/workflows/traffic.yml"
fi
