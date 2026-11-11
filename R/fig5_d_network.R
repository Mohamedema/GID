# Figure 5d: RB1 and its strongest altered partners
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_panel(rb1, "e", "fig5_d_network", FIG5$e)
