# Appendix Figure S7c: cell line scores of the two altered partners
source("R/setup.R")
source("R/lib_mutation.R")

smarca4 <- load_gene("SMARCA4")
export_panel(smarca4, "c", "supp7_c_cell_line_scores", SUPP7$c)
