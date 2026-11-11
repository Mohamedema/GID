# Figure 5c: cell line scores of the strongest partners
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_panel(rb1, "c", "fig5_c_cell_line_scores", FIG5$c)
