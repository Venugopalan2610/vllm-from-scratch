#!/usr/bin/env bash
# Current traffic (GitHub's rolling 14-day window) plus permanent counters.
R="${1:-Venugopalan2610/vllm-from-scratch}"
echo "== $R =="
gh api "repos/$R" --jq '"stars: \(.stargazers_count)   forks: \(.forks_count)   watchers: \(.subscribers_count)"'
echo
echo "last 14 days:"
gh api "repos/$R/traffic/clones" --jq '"  clones: \(.count) total, \(.uniques) unique"'
gh api "repos/$R/traffic/views"  --jq '"  views:  \(.count) total, \(.uniques) unique"'
echo
echo "top referrers:"
gh api "repos/$R/traffic/popular/referrers" --jq '.[] | "  \(.referrer): \(.count) (\(.uniques) unique)"' 2>/dev/null | head -5
echo
echo "who forked (permanent):"
gh api "repos/$R/forks" --jq '.[] | "  \(.owner.login)  \(.created_at[:10])"' 2>/dev/null | head -20
[ -f .traffic/history.json ] && echo && echo "accumulated history:" && python3 -c "
import json;h=json.load(open('.traffic/history.json'));s=h['summary']
print(f\"  clones: {s['clones']['total']} over {s['clones']['days_recorded']} days recorded\")
print(f\"  views:  {s['views']['total']} over {s['views']['days_recorded']} days recorded\")"
