#!/usr/bin/env bash
# PHASE 3 — pre-registration. Commit the frozen plan, quote the hash in the paper.
# Put predictions/thresholds/title-table/analysis-plan in prereg.md FIRST, then run this.
set -e

# thresholds you are committing to (these go in prereg.md):
#   Spearman >= 0.8 (TEST fold) ; DCC CI > 0 ; MAF CI > 0 ; MAF > 0.5 = "primarily"
#   cos(V3,V4) decision threshold ~0.8
#   predictions P1, P2, P3a, P3b ; D1..D10 frozen ; title decision table

git add prereg.md D*_frozen.json
git commit -m "Pre-registration: predictions, thresholds, frozen datasets, title table"
HASH=$(git rev-parse HEAD)
echo "PRE-REGISTRATION HASH (quote this in the report): $HASH"
echo "$HASH" > PREREG_HASH.txt

# optional public timestamp: push to GitHub now so the commit time is independently visible.
# git push origin main
