# Figure 5f: effect with and without the tissue term
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_panel(rb1, "g", "fig5_f_with_without_tissue", FIG5$g)
