# Appendix Figure S7a: SMARCA4 mutated cell lines per tissue
source("R/setup.R")
source("R/lib_mutation.R")

smarca4 <- load_gene("SMARCA4")
export_panel(smarca4, "a", "supp7_a_mutated_lines", SUPP7$a)
