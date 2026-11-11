# Figure 5a: mutated cell lines per tissue
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_panel(rb1, "a", "fig5_a_mutated_lines", FIG5$a)
