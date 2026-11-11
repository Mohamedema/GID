# Figure 2a: ROC curves for recovery of CORUM complex co-membership
source("R/setup.R")
source("R/lib_fig2.R")

p_a <- roc_plot("corum") + labs(tag = "a")

save_panel(p_a + theme(aspect.ratio = 1), "fig2_a_roc_corum", COL_L, ROW_H)
