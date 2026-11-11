# Figure 4b: density of tissue-specific complexes across tissues (row-wise z-score)
source("R/setup.R")
source("R/lib_heatmap.R")
library(readr)

sel            <- read.csv(file.path(results_dir, "Final_Rigorously_Tissue_Specific_Complexes.csv"),
                           stringsAsFactors = FALSE, check.names = FALSE)
tissue_summary <- read_csv(file.path(results_dir, "tissue_summary.csv"), show_col_types = FALSE)

# tissues with at least five cell lines, largest first
col_order <- tissue_summary %>%
  filter(num_samples >= 5, tissue %in% sel$Tissue) %>%
  arrange(desc(num_samples)) %>%
  pull(tissue)
sel <- filter(sel, Tissue %in% col_order)
z   <- density_z(file.path(results_dir, "complex_tissue_density_matrix.csv"), col_order)

# TNF-alpha/NF-kappa B signaling complex 7 (haematopoietic) and TLE1 corepressor complex (CNS)
manual <- data.frame(Tissue = c("Haematopoietic and Lymphoid", "Central Nervous System"),
                     Complex_ID_in_Model = c("Complex_1427", "Complex_1242"))

picks <- pick_complexes(sel, z, manual, "Complex_ID_in_Model", col_order)
hm    <- complex_heatmap(picks, z, col_order)

save_drawn(hm$draw, "fig4_b_heatmap", FIG_W_IN, hm$height)
save_drawn(draw_heatmap_legend, "fig4_b_legend", 3.5, 3)
