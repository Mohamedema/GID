# Appendix Figure S8b: density of subtype-specific complexes across the PAM50 subtypes
source("R/setup.R")
source("R/lib_heatmap.R")

sel       <- read.csv(file.path(results_dir, "tcga", "Final_Rigorously_Tissue_Specific_Complexes.csv"),
                      stringsAsFactors = FALSE, check.names = FALSE)
col_order <- c("Her2", "LumA", "LumB", "Basal")
z         <- density_z(file.path(results_dir, "tcga", "complex_tissue_density_matrix.csv"), col_order)

manual <- data.frame(Tissue = col_order,
                     complex_name = c("CCNC-CDK8-MED1-MED6-MED7 complex",
                                      "BASC (Ab 80) complex (BRCA1-associated genome surveillance complex)",
                                      "BCL3-ESR1-NCOA3-RELA complex",
                                      "c-MYC-ATPase-helicase complex"))

picks <- pick_complexes(sel, z, manual, "complex_name", col_order)
hm    <- complex_heatmap(picks, z, col_order)

save_drawn(hm$draw, "supp8_b_heatmap", FIG_W_IN, hm$height)
save_drawn(draw_heatmap_legend, "supp8_b_legend", 3.5, 3)
