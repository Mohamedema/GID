# Figure 2d: Jaccard index of tissue network pairs, by reference resource
source("R/setup.R")
source("R/lib_fig2.R")

jac <- read.csv(file.path(results_dir, "tissue_jaccard_pairs_top60percent.csv"))
jac <- jac[!is.na(jac$jaccard_index), ]

db_class <- c(CORUM = "Curated / annotated databases", STRING = "Curated / annotated databases",
              Reactome = "Curated / annotated databases", SIGNOR = "Curated / annotated databases",
              BioPlex = "Experimental interactome screens", HuRI = "Experimental interactome screens",
              Quantified = "Quantified")
class_colours <- c("Curated / annotated databases" = "#E64B35",
                   "Experimental interactome screens" = "#4DBBD5",
                   "Quantified" = "#B09C85")

db_order <- names(db_class)[names(db_class) %in% jac$database]
jac <- jac[jac$database %in% db_order, ]
jac$database <- factor(jac$database, levels = db_order)
jac$class    <- factor(db_class[as.character(jac$database)], levels = names(class_colours))

p_d <- ggplot(jac, aes(x = database, y = jaccard_index, fill = class)) +
  geom_violin(width = 0.9, trim = FALSE, colour = "black", linewidth = lw(0.6)) +
  geom_boxplot(width = 0.13, fill = "white", colour = "black", outlier.shape = NA, linewidth = lw(0.5)) +
  scale_x_discrete(limits = rev(db_order)) +
  scale_fill_manual(values = class_colours, name = NULL) +
  coord_flip() +
  labs(x = NULL, y = "Jaccard index", tag = "d") +
  theme_classic(base_size = FS_TITLE, base_family = "sans") +
  theme(axis.text.y       = element_text(colour = "black", size = FS_TITLE, face = "bold"),
        axis.text.x       = element_text(colour = "black", size = FS_TEXT),
        axis.title.x      = element_text(size = FS_TITLE),
        axis.line         = element_line(linewidth = lw(0.8), colour = "black"),
        axis.ticks        = element_line(linewidth = lw(0.8), colour = "black"),
        axis.ticks.length = unit(0.09, "cm"),
        legend.background = element_blank(),
        legend.key        = element_rect(fill = NA, colour = NA),
        legend.key.size   = unit(0.30, "cm"),
        legend.text       = element_text(size = FS_TEXT),
        plot.margin       = margin(10, 6, 2, 2),
        plot.tag          = element_text(size = FS_TAG, face = "bold", hjust = 0, vjust = 1),
        plot.tag.position = c(0, 1)) +
  legend_inside(c(0.99, 0.02), c("right", "bottom")) +
  guides(fill = guide_legend(override.aes = list(colour = "black", linewidth = lw(0.6))))

save_panel(p_d + theme(aspect.ratio = 1), "fig2_d_jaccard", COL_R + 0.5, ROW_H)
