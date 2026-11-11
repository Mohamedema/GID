# Figure 5h: altered edges when the mutation status is shuffled within tissue
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_panel(rb1, "i", "fig5_h_permutations", FIG5$i)
