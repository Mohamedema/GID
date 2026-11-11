# Appendix Figure S1: ROC curves for recovery of STRING edges
source("R/setup.R")
source("R/lib_fig2.R")

p <- roc_plot("string")

save_panel(p + theme(aspect.ratio = 1), "supp1_roc_string", COL_L, ROW_H)
