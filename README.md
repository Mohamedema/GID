# GID: Graph Inferred Deconvolution

GID infers sample-specific protein-protein association (PPA) networks from
multi-omics cancer cell line data.

## Overview

GID is a graph convolutional network (GCN) autoencoder that integrates five
omics layers — transcriptomics, proteomics, DNA methylation, copy number
variation, and CRISPR-Cas9 gene essentiality over a shared graph topology
derived from the STRING protein-protein association network. The model
encodes each gene's multi-omics profile into a low-dimensional latent
representation and reconstructs it with a mirrored graph decoder.

Population-level embeddings from the trained model are deconvolved into
individualized networks using a LIONESS-based leave one out procedure,
yielding one PPA network per sample rather than a single population or
tissue-level average. This repository was used to infer sample specific
networks for 1,007 cancer cell lines spanning 28 tissue types from the
Cancer Dependency Map.

![image_alt]https://github.com/Mohamedema/GID/blob/main/adjusted_figure1%20(4320%20x%204016%20px)%20(9).png

## Repository structure

```
.
├── src/                  # Model definition, training loop, data loading
├── scripts/              # Baseline comparisons and downstream analyses
└── environment.yml
```



Requires Python >=3.9, PyTorch, and PyTorch Geometric.

## Usage

Training and inference are configured via a single config file:

```bash
python src/main.py
```

## Data

GID requires:
- Multi-omics profiles per gene per sample (transcriptomics, proteomics,
  methylation, CNV, CRISPR essentiality), aligned to a common gene set.
- A STRING-derived protein-protein association network over the same
  gene set, used as the fixed graph topology.


