#!/usr/bin/env bash
# Every figure panel that can be drawn from the tables in results/.
set -euo pipefail

for script in R/fig*.R R/supp*.R; do
    Rscript "$script"
done

for script in python/*.py; do
    python "$script"
done
