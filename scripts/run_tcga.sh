#!/usr/bin/env bash
# TCGA-BRCA validation: two omic layers, PAM50 subtypes in place of tissues.
#   RAW=data/raw DEVICE=cuda bash scripts/run_tcga.sh
set -euo pipefail

raw=${RAW:-data/raw}
device=${DEVICE:-cpu}
config=${CONFIG:-configs/tcga_brca.json}

data=data/tcga_brca
runs=runs/tcga_brca
results=results/tcga

python -m gid.tcga --raw "$raw/tcga_brca" --string "$raw" --out "$data"
python -m gid.train --data "$data" --out "$runs" --config "$config" --device "$device"
python -m gid.infer --data "$data" --model "$runs/model.pt" --out "$runs" --device "$device"
python -m gid.complexes --networks "$runs/tissue_networks.npy" --names "$runs/tissue_names.txt" \
    --genes "$data/genes.txt" --corum "$raw/corum_humanComplexes.txt" --out "$results"
python -m gid.pooling --data "$data" --model "$runs/model.pt" --out "$results"
