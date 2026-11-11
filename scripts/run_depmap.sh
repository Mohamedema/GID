#!/usr/bin/env bash
# DepMap analysis, from the raw files to the tables behind the figures.
#   RAW=data/raw DEVICE=cuda bash scripts/run_depmap.sh
set -euo pipefail

raw=${RAW:-data/raw}
device=${DEVICE:-cpu}
config=${CONFIG:-configs/depmap_gid.json}

data=data/depmap
runs=runs/depmap
results=results
net=$runs/all_omics

python -m gid.preprocess --raw "$raw" --out "$data"

# the single omic models only need the population network and the clustering indices
for layer in mrna protein crispr methylation cnv; do
    python -m gid.train --data "$data" --out "$runs/$layer" --layers "$layer" --config "$config" --device "$device"
    python -m gid.infer --data "$data" --model "$runs/$layer/model.pt" --out "$runs/$layer" --device "$device"
    python -m gid.evaluate --data "$data" --population "$runs/$layer/population_network.npy" \
        --corum "$raw/corum_humanComplexes.txt" --name "$layer" --out "$results"
    python -m gid.resolution --data "$data" --networks "$runs/$layer/sample_networks.npy" --name "$layer" --out "$results"
    rm "$runs/$layer/sample_networks.npy"
done

python -m gid.train --data "$data" --out "$net" --config "$config" --device "$device"
python -m gid.infer --data "$data" --model "$net/model.pt" --out "$net" --device "$device"
python -m gid.evaluate --data "$data" --population "$net/population_network.npy" \
    --corum "$raw/corum_humanComplexes.txt" --name all_omics --out "$results"
python -m gid.resolution --data "$data" --networks "$net/sample_networks.npy" --name all_omics \
    --anatomy data/tissue_anatomy.csv --similarity --out "$results"

python -m gid.association --data "$data" --networks "$net/sample_networks.npy" --out "$results"
python -m gid.complexes --networks "$net/tissue_networks.npy" --names "$net/tissue_names.txt" \
    --genes "$data/genes.txt" --corum "$raw/corum_humanComplexes.txt" --out "$results"
python -m gid.conservation --data "$data" --networks "$net/tissue_networks.npy" --names "$net/tissue_names.txt" \
    --references "$raw" --out "$results"
python -m gid.benchmark --data "$data" --networks "$net/sample_networks.npy" --out "$results/benchmark"
python -m gid.pooling --data "$data" --model "$net/model.pt" --out "$results"

python -m gid.validation intact --data "$data" --file "$raw/Cancer.txt" --networks "$net/tissue_networks.npy" \
    --names "$net/tissue_names.txt" --tissue "Haematopoietic and Lymphoid" --out "$results"
python -m gid.validation synlethdb --data "$data" --file "$raw/Human.SL.detailed.tsv" \
    --population "$net/population_network.npy" --all-cells --suffix _SynLethDB_population --out "$results"

python -m gid.mutations --data "$data" --networks "$net/sample_networks.npy" \
    --genomics "$raw/cell_line_mutations.csv" --targets "$raw/open_targets_leukaemia.tsv" --out "$results/mutations"

python -m gid.seed_network --data "$data" --networks "$net/tissue_networks.npy" --names "$net/tissue_names.txt" \
    --targets "$raw/open_targets_leukaemia.tsv" --out "$results"
