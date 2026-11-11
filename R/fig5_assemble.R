# Figure 5: panels a to h on one page, 180 mm wide
source("R/setup.R")
source("R/lib_mutation.R")

rb1 <- load_gene("RB1")
export_page(rb1, FIG5, "fig5_abcdefgh")
