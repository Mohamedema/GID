# Appendix Figure S6: effect of RB1 mutation within individual tissues
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_panel(rb1, "d", "supp6_rb1_heatmap", SUPP6$d)
