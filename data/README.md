# Data

`tissue_anatomy.csv` is the only table kept in the repository. It assigns the 28 tissue types of the Cell Model Passports
annotation to 11 organ systems and is used for the organ system labels of the benchmark and of Figure 2 and Appendix
Figure S2. Everything else is downloaded or written by the pipeline.

The downloaded files go into `data/raw/` under the names below. The names are fixed in `gid/preprocess.py`,
`gid/tcga.py` and `gid/references.py`; a different name can be passed to the functions in Python, but the command line
scripts expect these.

## Training data (`gid.preprocess`)

| File | Content | Source |
|---|---|---|
| `mmc3.xlsx` | Proteomics of the cell lines (second sheet: cell lines in rows, `label;UniProt accession` in columns) | Supplementary table of Gonçalves et al., 2022 |
| `idmapping_2024_07_29.tsv` | UniProt accession and gene symbol, two columns, no header | UniProt ID mapping |
| `model_list_20230517.csv` | Cell line annotation, with the columns `model_id` and `tissue` | Cell Model Passports |
| `mRNA.csv` | Gene expression, genes in rows and cell lines in columns | Cell Model Passports |
| `Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_fold_change_values_20240111.tsv` | CRISPR-Cas9 fold changes | Project Score, Cell Model Passports |
| `gene_cnv_2018-11-13_1545.csv` | Gene level copy number in long format (`model_id`, `gene_symbol`, `gistic_score`) | Cell Model Passports |
| `methylation.csv` | Methylation, genes in rows and cell lines in columns | Cell Model Passports |
| `9606.protein.links.v12.0.txt` and `9606.protein.info.v12.0.txt` | STRING 12.0 combined scores and protein names | STRING |

Alternative prior network: `hippie_prior.tsv`, a tab separated edge list with the columns `gene_a`, `gene_b` and `score`
(gene symbols, one pair per row). It replaces STRING when it is passed to `gid.preprocess` with `--prior`.

## Reference resources (`gid.evaluate`, `gid.complexes`, `gid.conservation`, `gid.validation`)

| File | Content | Source |
|---|---|---|
| `corum_humanComplexes.txt` | Human complexes; the column `subunits_gene_name` lists the subunits separated by `;` | CORUM |
| `BioPlex3.tsv`, `signor_human.tsv`, `HuRI.tsv`, `reactome_interactions.txt` | Interaction resources compared with the tissue networks in `gid.conservation`; each one is used when present | BioPlex 3.0, SIGNOR, HuRI, Reactome |
| `hgnc_map.txt` | HGNC table with the columns `symbol` and `ensembl_gene_id`, needed to read HuRI and Reactome | HGNC |
| `Cancer.txt` | IntAct cancer dataset as a PSI-MI TAB 2.7 export | IntAct |
| `Human.SL.detailed.tsv` | Human synthetic lethal pairs | SynLethDB |
| `open_targets_leukaemia.tsv` | Associated targets of leukaemia (EFO_0000565), tab separated, with at least the columns `symbol`, `globalScore`, `gwasCredibleSets`, `hasSmallMoleculeBinder`, `maxClinicalStage` and `isCancerDriverGene` | Open Targets Platform |
| `cell_line_mutations.csv` | Somatic mutation status, genes in rows and cell lines (`SIDM` identifiers) in columns, 1 for mutated and 0 for wild type | Cell Model Passports |

## TCGA-BRCA (`gid.tcga`)

The three files go into `data/raw/tcga_brca/`; STRING is read from `data/raw/`.

| File | Content | Source |
|---|---|---|
| `fpkm_data.csv` | mRNA expression (FPKM) of the tumours, tumours in rows | Processed BRCA data of the MoGCN study, from UCSC Xena |
| `gistic_data.csv` | Gene level copy number (GISTIC), tumours in rows | as above |
| `sample_classes.csv` | PAM50 subtype of each tumour, with the columns `Sample_ID` and `class` | as above |

## Written by the pipeline

| Folder | Written by | Content |
|---|---|---|
| `data/depmap/`, `data/hippie/`, `data/tcga_brca/` | `gid.preprocess`, `gid.tcga` | `values.npy` (samples, layers, genes), `mask.npy`, `adjacency.npy`, `genes.txt`, `samples.csv`, `layers.json` |
| `runs/<name>/` | `gid.train`, `gid.infer` | `model.pt`, `training_loss.csv`, `population_network.npy`, `sample_networks.npy`, `tissue_networks.npy`, `tissue_names.txt` |
| `results/` | the analysis modules | the tables that the scripts in `R/` and `python/` read |

`sample_networks.npy` holds the upper triangle of the network of every sample as 32 bit floats, so it takes
samples x genes x (genes - 1) / 2 x 4 bytes: 8 GB for the 1,007 cell lines (1,998 genes) and 26 GB for the 511 tumours
(5,011 genes).

## Tables in `results/` and the figures that use them

| Table | Written by | Used for |
|---|---|---|
| `<layer>_roc_corum.csv`, `<layer>_roc_string.csv` (`mrna`, `protein`, `crispr`, `methylation`, `cnv`, `all_omics`) | `gid.evaluate` | Figure 2a, Appendix Figure S1 |
| `<layer>_clustering_metrics.csv` | `gid.resolution` | Figure 2b |
| `tissue_similarity.csv`, `anatomy_similarity.csv` | `gid.resolution --similarity` | Figure 2c, Appendix Figure S2b |
| `tissue_jaccard_pairs_top60percent.csv` | `gid.conservation` | Figure 2d |
| `benchmark/clustering_all_results_long.csv` | `gid.benchmark` | Figure 3 |
| `tissue_summary.csv`, `sharing_distribution.csv` | `gid.association` | Figure 4a |
| `Final_Rigorously_Tissue_Specific_Complexes.csv`, `complex_tissue_density_matrix.csv` | `gid.complexes` | Figure 4b |
| `mutations/<GENE>/`, `mutations/all_genes_summary.csv` | `gid.mutations` | Figure 5, Appendix Figures S5 to S7 |
| `gid_enrichment_data_with_null.csv`, `gid_enrichment_stats.csv` | `gid.validation intact` | Figure 6b |
| `gprofiler_first_degree.csv` | g:Profiler, from `*_first_degree_genes.txt` of `gid.seed_network` | Figure 6c |
| `gid_enrichment_data_with_null_SynLethDB_population.csv`, `gid_enrichment_stats_SynLethDB_population.csv` | `gid.validation synlethdb` | Appendix Figure S3 |
| `hippie/` (same layout as the main results) | `scripts/run_hippie.sh` | Appendix Figure S4 |
| `tcga/Final_Rigorously_Tissue_Specific_Complexes.csv`, `tcga/complex_tissue_density_matrix.csv` | `gid.complexes` on the TCGA networks | Appendix Figure S8 |
| `umap_samples.csv`, `tcga/umap_samples.csv` | `gid.pooling` | Appendix Figure S9 |
