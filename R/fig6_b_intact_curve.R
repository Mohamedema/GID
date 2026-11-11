# Figure 6b: IntAct interactions captured among the highest weighted edges of the population level network
source("R/setup.R")
library(scales)
library(cowplot)

curve <- read.csv(file.path(results_dir, "gid_enrichment_data_with_null.csv"))
stats <- read.csv(file.path(results_dir, "gid_enrichment_stats.csv"), stringsAsFactors = FALSE)
top5  <- stats[stats$top_pct == 5, ]
note  <- paste0("Top ", top5$top_pct, "%: ", sprintf("%.1f", top5$fold), " fold enrichment\n", top5$p_label)

p <- ggplot(curve, aes(x = Proportion_Edges_Ranked)) +
  geom_ribbon(aes(ymin = Null_CI_Lower, ymax = Null_CI_Upper, fill = "Null 95% CI"), alpha = 0.3) +
  geom_ribbon(aes(ymin = Model_CI_Lower, ymax = Model_CI_Upper, fill = "Model 95% CI"), alpha = 0.25) +
  geom_line(aes(y = Random_Expectation, linetype = "Random Expectation"), colour = "#4D4D4D", linewidth = 0.8) +
  geom_line(aes(y = Model_Capture_Rate, colour = "GID Model"), linewidth = 1.2) +
  scale_x_continuous(labels = percent_format(accuracy = 1), expand = c(0, 0), limits = c(0, 0.20)) +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = c(0, 0),
                     limits = c(0, max(curve$Model_CI_Upper) + 0.05)) +
  scale_fill_manual(name = NULL, values = c("Model 95% CI" = "#B22222", "Null 95% CI" = "#A9A9A9")) +
  scale_colour_manual(name = NULL, values = c("GID Model" = "#B22222")) +
  scale_linetype_manual(name = NULL, values = c("Random Expectation" = "dashed")) +
  labs(x = "Proportion of Network Edges Examined (Highest to Lowest Weight)",
       y = "Proportion of IntAct Interactions Captured") +
  annotate("text", x = 0.02, y = max(curve$Model_CI_Upper), label = note,
           hjust = 0, vjust = 1, fontface = "bold", size = 4.5, colour = "black") +
  theme_classic(base_size = 12) +
  theme(panel.background = element_rect(fill = "white", colour = NA),
        plot.background = element_rect(fill = "white", colour = NA),
        axis.text = element_text(colour = "black", size = 11),
        axis.title = element_text(face = "bold", size = 12, margin = margin(t = 10, r = 10)),
        axis.line = element_line(linewidth = 0.6, colour = "black"),
        axis.ticks = element_line(colour = "black", linewidth = 0.6),
        legend.position = "bottom", legend.direction = "horizontal",
        legend.background = element_rect(fill = "white", colour = NA),
        legend.box.background = element_rect(fill = "white", colour = NA),
        legend.key = element_blank(),
        legend.text = element_text(size = 11)) +
  guides(colour = guide_legend(order = 1),
         fill = guide_legend(order = 2, override.aes = list(alpha = 0.3)),
         linetype = guide_legend(order = 3))

# the plot and its legend are saved as two files
save_panel(p + theme(legend.position = "none"), "fig6_b_intact_curve", 6.5, 5)
save_panel(plot_grid(get_legend(p)), "fig6_b_intact_legend", 6.5, 1)
