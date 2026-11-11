# Figure 5g: effect with the heaviest tissue left out
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_panel(rb1, "h", "fig5_g_heaviest_tissue_left_out", FIG5$h)
