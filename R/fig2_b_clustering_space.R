# Figure 2b: Calinski-Harabasz index against 1 / Davies-Bouldin index, tissue and organ level
source("R/setup.R")
source("R/lib_fig2.R")
library(ggrepel)

files <- c("Transcriptomics" = "mrna", "Proteomics" = "protein", "CRISPR-Cas9" = "crispr",
           "Methylation" = "methylation", "CNV" = "cnv", "Multi-Omics" = "all_omics")

clust <- bind_rows(lapply(names(files), function(o) {
  d <- read.csv(file.path(results_dir, paste0(files[[o]], "_clustering_metrics.csv")))
  d$Omic <- o
  d
}))
clust$Omic <- factor(clust$Omic, levels = omic_order)

long <- clust %>%
  select(-Tag) %>%
  pivot_longer(-Omic, names_to = "metric", values_to = "value") %>%
  filter(grepl("Continuous", metric)) %>%
  mutate(index = ifelse(grepl("^CH", metric), "CH", "DB"),
         Condition = factor(ifelse(grepl("Tissue", metric), "Tissue", "Organ"), levels = c("Tissue", "Organ")))

perf <- left_join(long %>% filter(index == "CH") %>% select(Omic, Condition, CH = value),
                  long %>% filter(index == "DB") %>% transmute(Omic, Condition, DB_inv = 1 / value),
                  by = c("Omic", "Condition"))

p_b <- ggplot(perf, aes(x = CH, y = DB_inv)) +
  annotate("rect", xmin = median(perf$CH), xmax = max(perf$CH) * 1.05,
           ymin = median(perf$DB_inv), ymax = max(perf$DB_inv) * 1.05,
           fill = "#E8F5E9", alpha = 0.45, colour = NA) +
  geom_line(aes(group = Omic, colour = Omic), linewidth = lw(0.7), alpha = 0.5, linetype = "dashed") +
  geom_point(aes(colour = Omic, shape = Condition), size = 2.3, alpha = 0.95, stroke = 0.5) +
  geom_text_repel(aes(label = paste0(Omic, "\n(", Condition, ")"), colour = Omic),
                  size = pt2mm(FS_TEXT), lineheight = 0.85, max.overlaps = Inf,
                  box.padding = 0.35, point.padding = 0.25, min.segment.length = 0.2,
                  segment.color = "grey60", segment.size = lw(0.4),
                  bg.color = "white", bg.r = 0.12, seed = 1, show.legend = FALSE) +
  scale_colour_manual(name = "Omic", values = omic_colours) +
  scale_shape_manual(name = "Level", values = c("Tissue" = 16, "Organ" = 17),
                     labels = c("Tissue \u2013 Continuous", "Organ \u2013 Continuous")) +
  labs(x = "Calinski-Harab\u00e1sz Index (higher = better separation)",
       y = "1 / Davies-Bouldin Index (higher = better cohesion)", tag = "b") +
  guides(colour = guide_legend(ncol = 1, override.aes = list(size = 2.3), order = 1),
         shape  = guide_legend(ncol = 1, override.aes = list(size = 2.3), order = 2)) +
  theme_nature() +
  legend_inside(c(0.99, 0.02), c("right", "bottom")) +
  theme(legend.box = "vertical", legend.box.just = "left",
        legend.spacing.y = unit(0.08, "cm"),
        legend.margin = margin(2, 4, 2, 4),
        legend.key.size = unit(0.28, "cm"))

save_panel(p_b + theme(aspect.ratio = 1), "fig2_b_clustering_space", COL_L, ROW_H)
