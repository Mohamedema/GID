# Figure 5b: RB1 associations, mutated against wild type
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_panel(rb1, "b", "fig5_b_volcano", FIG5$b)
