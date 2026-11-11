# Appendix Figure S7b: SMARCA4 associations, mutated against wild type
source("R/setup.R")
source("R/lib_mutation.R")

smarca4 <- load_gene("SMARCA4")
export_panel(smarca4, "b", "supp7_b_volcano", SUPP7$b)
