import sys
from unittest import mock

import numpy as np
import pandas as pd

from gid import cytoscape_draw


def targets():
    return pd.DataFrame({
        "symbol": ["tp53", "ESR1", "NCOA3", "BCL3", "RELA"],
        "isCancerDriverGene": ["1", "no", "0", "nan", "yes"],
        "hasSmallMoleculeBinder": ["false", "true", "0", "0", "0"],
        "hasPocket": ["0", "0", "1.0", "0", "0"],
        "maxClinicalStage": [4, 3, np.nan, 0, 1],
        "globalScore": [0.9, 0.7, 0.2, 0.1, 0.4],
    })


def test_annotations_follow_the_columns_of_the_table():
    ann = cytoscape_draw.OpenTargets(targets())
    assert ann.drivers == {"TP53", "RELA"}
    assert ann.druggable == {"ESR1", "NCOA3"}
    assert ann.genes == {"TP53", "ESR1", "NCOA3", "BCL3", "RELA"}
    assert ann.clinical_stage["TP53"] == 4 and np.isnan(ann.clinical_stage["NCOA3"])


def test_node_categories_in_order_of_precedence():
    ann = cytoscape_draw.OpenTargets(targets())
    nodes = pd.DataFrame({
        "Node": ["ESR1", "TP53", "NCOA3", "BCL3", "XYZ1", "rs123", "RELA"],
        "Node_Type": ["Core_Complex", "STRING_Extended", "STRING_Extended", "STRING_Extended", "STRING_Extended",
                      "STRING_Extended", "STRING_Extended"],
    })
    out = cytoscape_draw.annotate_nodes(nodes, ann)
    assert out["Clinical_Category"].tolist() == ["Core_Complex", "Cancer_Driver", "Drug_Target", "Disease_Associated",
                                                 "Other", "SNP", "Cancer_Driver"]
    assert "Node" not in out.columns and "gene_upper" not in out.columns
    assert out.set_index("id").loc["XYZ1", "globalScore"] == 0
    assert out.set_index("id").loc["TP53", "globalScore"] == 0.9
    assert out.set_index("id").loc["BCL3", "maxClinicalStage"] == 0


def test_edges_take_the_first_available_change_column():
    edges = pd.DataFrame({"Source": ["a", "b"], "Target": ["b", "c"], "Delta_Home_vs_PanCancer": [-0.3, 0.2],
                          "Delta": [9, 9], "Known_Interaction": [1, 0], "Edge_Color_Cytoscape": ["Blue", "Red"]})
    out, delta = cytoscape_draw.prepare_edges(edges)
    assert delta == "Delta_Home_vs_PanCancer"
    assert out["Abs_Delta"].tolist() == [0.3, 0.2]
    assert out["Known_Interaction"].tolist() == ["1", "0"]
    assert out["Edge_Category"].tolist() == ["Blue", "Red"]
    assert {"source", "target"} <= set(out.columns)


def test_edges_without_a_change_column_get_default_values():
    out, delta = cytoscape_draw.prepare_edges(pd.DataFrame({"Source": ["a"], "Target": ["b"]}))
    assert delta is None and out["Abs_Delta"].tolist() == [1.0] and out["Edge_Category"].tolist() == ["Gray"]


def test_the_network_is_built_before_it_is_styled():
    fake = mock.MagicMock()
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"], "Abs_Delta": [0.2, 0.6], "Known_Interaction": ["1", "0"]})
    with mock.patch.dict(sys.modules, {"py4cytoscape": fake}):
        cytoscape_draw.draw(pd.DataFrame({"id": ["a", "b", "c"]}), edges, "Delta", "t")
    names = [c[0] for c in fake.method_calls]
    assert names[0] == "create_network_from_data_frames" and names[-1] == "layout_network"
    width = fake.set_edge_line_width_mapping.call_args.kwargs
    assert width["table_column_values"] == [0.0, 0.3, 0.6] and width["widths"] == [1.5, 4.5, 8.0]
    assert fake.set_edge_opacity_mapping.call_args.kwargs["opacities"] == [120, 255]
