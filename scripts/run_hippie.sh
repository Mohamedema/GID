#!/usr/bin/env bash
# The same model with HIPPIE instead of STRING as the prior network.
#   RAW=data/raw DEVICE=cuda bash scripts/run_hippie.sh
set -euo pipefail

raw=${RAW:-data/raw}
device=${DEVICE:-cpu}
config=${CONFIG:-configs/depmap_gid.json}

data=data/hippie
runs=runs/hippie
results=results/hippie

python -m gid.preprocess --raw "$raw" --prior "$raw/hippie_prior.tsv" --out "$data"
python -m gid.train --data "$data" --out "$runs" --config "$config" --device "$device"
python -m gid.infer --data "$data" --model "$runs/model.pt" --out "$runs" --device "$device"
python -m gid.benchmark --data "$data" --networks "$runs/sample_networks.npy" --out "$results/benchmark"
python -m gid.complexes --networks "$runs/tissue_networks.npy" --names "$runs/tissue_names.txt" \
    --genes "$data/genes.txt" --corum "$raw/corum_humanComplexes.txt" --out "$results"
python -m gid.mutations --data "$data" --networks "$runs/sample_networks.npy" \
    --genomics "$raw/cell_line_mutations.csv" --out "$results/mutations"
