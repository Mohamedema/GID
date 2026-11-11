# Appendix Figure S9a: UMAP of the sample embeddings of the cell lines, coloured by tissue
source("R/setup.R")

umap <- read.csv(file.path(results_dir, "umap_samples.csv"))

# 28 tissues in alphabetical order, one IGV colour each
tissues <- sort(unique(umap$Tissue))
tissue_colours <- setNames(ggsci::pal_igv("default")(length(tissues)), tissues)

p <- ggplot(umap, aes(x = UMAP1, y = UMAP2, fill = Tissue)) +
  geom_point(shape = 21, size = 2.8, colour = "white", stroke = 0.3, alpha = 0.9) +
  scale_fill_manual(values = tissue_colours, name = NULL) +
  labs(x = "UMAP 1", y = "UMAP 2") +
  guides(fill = guide_legend(override.aes = list(size = 3.5, stroke = 0.3), ncol = 2)) +
  theme_classic(base_size = 11) +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        axis.line = element_line(colour = "black", linewidth = 0.8),
        axis.title = element_text(face = "bold", size = 12, colour = "black"),
        legend.title = element_blank(),
        legend.text = element_text(size = 8, colour = "black"),
        legend.key = element_blank(), legend.key.size = unit(0.4, "cm"),
        legend.background = element_blank(),
        plot.title = element_blank(),
        panel.background = element_rect(fill = "white", colour = NA),
        plot.background = element_rect(fill = "white", colour = NA))

save_panel(p, "supp9_a_umap_tissue", 12, 8)
