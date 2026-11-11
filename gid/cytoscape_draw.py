import argparse
import os

import pandas as pd

NOT_A_DRIVER = ["0", "0.0", "nan", "none", "false", "no"]
DRUGGABLE_COLUMNS = ["hasSmallMoleculeBinder", "hasPocket", "hasHighQualityChemicalProbes"]
DELTA_COLUMNS = ["Delta_Home_vs_Other", "Delta_Home_vs_PanCancer", "Delta"]
OT_COLUMNS = ["globalScore", "gwasCredibleSets", "hasSafetyEvent", "isInMembrane", "hasLigand", "tissueSpecificity"]

CATEGORIES = ["Core_Complex", "Cancer_Driver", "Drug_Target", "Disease_Associated", "Other", "SNP"]
CATEGORY_COLOURS = ["#4CAF50", "#E63946", "#2CA02C", "#2B7FBF", "#D3D3D3", "#C8E6C9"]
EDGE_CATEGORIES = ["Purple", "Gold", "Gray", "Red", "Blue"]
EDGE_COLOURS = ["#D6B4FC", "#F4D068", "#C8C8C8", "#E63946", "#2B7FBF"]


class OpenTargets:
    """Gene level annotations of an Open Targets associated targets table."""

    def __init__(self, table):
        table = table.copy()
        self.symbol = "symbol" if "symbol" in table.columns else table.columns[0]
        table[self.symbol] = table[self.symbol].astype(str).str.strip().str.upper()
        self.table = table
        driver = table["isCancerDriverGene"].apply(lambda v: str(v).strip().lower() not in NOT_A_DRIVER)
        self.drivers = set(table.loc[driver, self.symbol])
        self.druggable = set(table.loc[table.apply(self.is_druggable, axis=1), self.symbol])
        self.genes = set(table[self.symbol])
        self.clinical_stage = {}
        if "maxClinicalStage" in table.columns:
            for gene, stage in zip(table[self.symbol], table["maxClinicalStage"]):
                try:
                    self.clinical_stage[gene] = float(stage)
                except (TypeError, ValueError):
                    pass

    @staticmethod
    def is_druggable(row):
        for col in DRUGGABLE_COLUMNS:
            if str(row.get(col, "")).strip().lower() in ["true", "1", "1.0", "yes"]:
                return True
        return False


def category(node, node_type, annotations):
    if str(node).strip().lower().startswith("rs"):
        return "SNP"
    gene = str(node).strip().upper()
    if node_type == "Core_Complex":
        return "Core_Complex"
    if gene in annotations.drivers:
        return "Cancer_Driver"
    if gene in annotations.druggable:
        return "Drug_Target"
    if gene in annotations.genes:
        return "Disease_Associated"
    return "Other"


def annotate_nodes(nodes, annotations):
    nodes = nodes.rename(columns={"Node": "id"})
    node_type = nodes["Node_Type"] if "Node_Type" in nodes.columns else pd.Series(None, index=nodes.index)
    nodes["Clinical_Category"] = [category(n, t, annotations) for n, t in zip(nodes["id"], node_type)]
    nodes["gene_upper"] = nodes["id"].str.strip().str.upper()
    nodes["maxClinicalStage"] = nodes["gene_upper"].map(annotations.clinical_stage).fillna(0)
    slim = annotations.table[[annotations.symbol] + [c for c in OT_COLUMNS if c in annotations.table.columns]]
    slim = slim.rename(columns={annotations.symbol: "gene_upper"})
    nodes = nodes.merge(slim, on="gene_upper", how="left").drop(columns=["gene_upper"])
    nodes["globalScore"] = nodes["globalScore"].fillna(0)
    return nodes


def prepare_edges(edges):
    """Edge table in the layout Cytoscape expects; returns it with the name of the column that holds the change."""
    edges = edges.rename(columns={"Source": "source", "Target": "target"})
    edges["Edge_Category"] = edges["Edge_Color_Cytoscape"] if "Edge_Color_Cytoscape" in edges.columns else "Gray"
    delta = next((c for c in DELTA_COLUMNS if c in edges.columns), None)
    edges["Abs_Delta"] = edges[delta].abs() if delta else 1.0
    if "Known_Interaction" in edges.columns:
        edges["Known_Interaction"] = edges["Known_Interaction"].astype(str)
    return edges, delta


def draw(nodes, edges, delta, title):
    import py4cytoscape as p4c

    p4c.create_network_from_data_frames(nodes=nodes, edges=edges, title=title, collection="Single Complex Analysis")
    p4c.set_current_network(title)

    style = "Single_Network_Style"
    defaults = {
        "NODE_SHAPE": "ellipse",
        "NODE_SIZE": 50,
        "NODE_BORDER_WIDTH": 2.0,
        "NODE_BORDER_PAINT": "#555555",
        "NODE_LABEL_POSITION": "W,E,c,5.00,0.00",
        "NODE_LABEL_FONT_SIZE": 13,
        "NODE_LABEL_FONT_FACE": "Arial,bold,italic",
        "NODE_LABEL_COLOR": "#000000",
        "EDGE_TRANSPARENCY": 210,
        "NETWORK_BACKGROUND_PAINT": "#FFFFFF",
    }
    labels = p4c.map_visual_property("node label", "id", "p")
    p4c.create_visual_style(style, defaults, [labels])
    p4c.set_visual_style(style, network=title)

    p4c.set_node_shape_mapping(table_column="Clinical_Category", table_column_values=["SNP"],
                               shapes=["ROUND_RECTANGLE"], style_name=style, network=title)
    p4c.set_node_color_mapping(table_column="Clinical_Category", table_column_values=CATEGORIES,
                               colors=CATEGORY_COLOURS, mapping_type="d", style_name=style, network=title)
    p4c.set_node_size_mapping(table_column="globalScore", table_column_values=[0.0, 0.5, 1.0], sizes=[38, 55, 75],
                              mapping_type="c", style_name=style, network=title)
    p4c.set_edge_color_mapping(table_column="Edge_Category", table_column_values=EDGE_CATEGORIES,
                               colors=EDGE_COLOURS, mapping_type="d", style_name=style, network=title)
    if delta:
        top = edges["Abs_Delta"].max()
        p4c.set_edge_line_width_mapping(table_column="Abs_Delta", table_column_values=[0.0, top / 2, top],
                                        widths=[1.5, 4.5, 8.0], mapping_type="c", style_name=style, network=title)
    if "Known_Interaction" in edges.columns:
        p4c.set_edge_opacity_mapping(table_column="Known_Interaction", table_column_values=["0", "1"],
                                     opacities=[120, 255], mapping_type="d", style_name=style, network=title)
    p4c.layout_network("force-directed", network=title)


def main():
    ap = argparse.ArgumentParser(description="Draw the network of one complex in a running Cytoscape session")
    ap.add_argument("--nodes", required=True, help="*_nodes.csv written by gid.cytoscape")
    ap.add_argument("--edges", required=True, help="*_edges.csv written by gid.cytoscape")
    ap.add_argument("--targets", required=True, help="Open Targets associated targets (tsv)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    title = args.title or "Single Network - " + os.path.basename(args.nodes).replace("_nodes.csv", "")
    annotations = OpenTargets(pd.read_csv(args.targets, sep="\t"))
    nodes = annotate_nodes(pd.read_csv(args.nodes), annotations)
    edges, delta = prepare_edges(pd.read_csv(args.edges))
    draw(nodes, edges, delta, title)


if __name__ == "__main__":
    main()
