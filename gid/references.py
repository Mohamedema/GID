import os
from itertools import combinations

import numpy as np
import pandas as pd

from .data import edge_column, upper_index

CORUM_FILE = "corum_humanComplexes.txt"
BIOPLEX_FILE = "BioPlex3.tsv"
SIGNOR_FILE = "signor_human.tsv"
HURI_FILE = "HuRI.tsv"
REACTOME_FILE = "reactome_interactions.txt"
HGNC_FILE = "hgnc_map.txt"


def gene_lookup(genes):
    return {g.strip().upper(): i for i, g in enumerate(genes)}


def pairs_mask(pairs, lookup, n_genes):
    """Boolean vector over the upper-triangle gene pairs, True for the pairs listed."""
    mask = np.zeros(n_genes * (n_genes - 1) // 2, dtype=bool)
    for a, b in pairs:
        a, b = str(a).strip().upper(), str(b).strip().upper()
        if a in lookup and b in lookup and a != b:
            i, j = sorted((lookup[a], lookup[b]))
            mask[edge_column(i, j, n_genes)] = True
    return mask


def read_hgnc(path):
    table = pd.read_csv(path, sep="\t", low_memory=False, dtype=str).fillna("")
    if not {"symbol", "ensembl_gene_id"} <= set(table.columns):
        return {}
    out = {}
    for ensg, symbol in zip(table["ensembl_gene_id"].str.strip(), table["symbol"].str.strip().str.upper()):
        if ensg and symbol:
            out[ensg] = symbol
    return out


def to_symbol(ensg, table):
    ensg = ensg.strip()
    return table.get(ensg, table.get(ensg.upper(), ""))


def ensembl_id(field):
    for token in field.split("|"):
        if "ENSG" in token:
            ens = token.split(":")[-1].strip()
            if ens.startswith("ENSG"):
                return ens
    return ""


def string_mask(adj, threshold=0.7):
    ii, jj = upper_index(adj.shape[0])
    return np.asarray(adj, dtype=np.float32)[ii, jj] >= threshold


def corum_mask(path, lookup, n_genes):
    corum = pd.read_csv(path, sep="\t", low_memory=False)
    pairs = []
    for members in corum["subunits_gene_name"].dropna():
        inside = [g.strip().upper() for g in str(members).split(";") if g.strip().upper() in lookup]
        pairs.extend(combinations(inside, 2))
    return pairs_mask(pairs, lookup, n_genes)


def bioplex_mask(path, lookup, n_genes):
    table = pd.read_csv(path, sep="\t", low_memory=False)
    if "pInt" in table.columns:
        table = table[table["pInt"] > 0.5]
    return pairs_mask(zip(table["SymbolA"], table["SymbolB"]), lookup, n_genes)


def signor_mask(path, lookup, n_genes):
    table = pd.read_csv(path, sep="\t", low_memory=False)
    table = table[(table["TYPEA"].str.upper() == "PROTEIN") & (table["TYPEB"].str.upper() == "PROTEIN")]
    return pairs_mask(zip(table["ENTITYA"], table["ENTITYB"]), lookup, n_genes)


def huri_mask(path, lookup, n_genes, hgnc):
    table = pd.read_csv(path, sep="\t", header=None, names=["ensg_a", "ensg_b"], low_memory=False)
    pairs = []
    for a, b in zip(table["ensg_a"], table["ensg_b"]):
        sym_a, sym_b = to_symbol(str(a), hgnc), to_symbol(str(b), hgnc)
        if sym_a and sym_b:
            pairs.append((sym_a, sym_b))
    return pairs_mask(pairs, lookup, n_genes)


def reactome_mask(path, lookup, n_genes, hgnc):
    pairs = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            ens_a = ensembl_id(parts[1])
            ens_b = ensembl_id(parts[4]) if len(parts) > 4 else ""
            if ens_a and ens_b:
                sym_a, sym_b = to_symbol(ens_a, hgnc), to_symbol(ens_b, hgnc)
                if sym_a and sym_b:
                    pairs.append((sym_a, sym_b))
    return pairs_mask(pairs, lookup, n_genes)


def reference_masks(genes, adj, folder, string_threshold=0.7):
    """Pairs of the network genes found in each reference resource that is present in `folder`."""
    n = len(genes)
    lookup = gene_lookup(genes)

    def path(name):
        return os.path.join(folder, name)

    masks = {"STRING": string_mask(adj, string_threshold)}
    if os.path.exists(path(CORUM_FILE)):
        masks["CORUM"] = corum_mask(path(CORUM_FILE), lookup, n)
    if os.path.exists(path(BIOPLEX_FILE)):
        masks["BioPlex"] = bioplex_mask(path(BIOPLEX_FILE), lookup, n)
    if os.path.exists(path(SIGNOR_FILE)):
        masks["SIGNOR"] = signor_mask(path(SIGNOR_FILE), lookup, n)
    hgnc = read_hgnc(path(HGNC_FILE)) if os.path.exists(path(HGNC_FILE)) else {}
    if hgnc and os.path.exists(path(HURI_FILE)):
        masks["HuRI"] = huri_mask(path(HURI_FILE), lookup, n, hgnc)
    if hgnc and os.path.exists(path(REACTOME_FILE)):
        masks["Reactome"] = reactome_mask(path(REACTOME_FILE), lookup, n, hgnc)
    return masks
