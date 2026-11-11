# Figure 5e: altered reference edges by degree from RB1
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_panel(rb1, "f", "fig5_e_degrees", FIG5$f)
