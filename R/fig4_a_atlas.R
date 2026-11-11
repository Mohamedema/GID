# Figure 4a: radial plot of likely and confident associations per tissue, and sharing across tissues
source("R/setup.R")
library(readr)

tissue_summary <- read_csv(file.path(results_dir, "tissue_summary.csv"), show_col_types = FALSE)
sharing        <- read_csv(file.path(results_dir, "sharing_distribution.csv"), show_col_types = FALSE)

atlas_colours <- c("Likely (>0.5)" = "#9ecae1", "Confident (>0.7)" = "#08519c")

tissue_summary <- tissue_summary %>%
  filter(num_samples >= 5) %>%
  arrange(desc(num_samples)) %>%
  mutate(tissue_label = paste0(tissue, " (", num_samples, ")"),
         tissue_id = row_number(),
         likely_log = log10(likely + 1),
         confident_log = log10(confident + 1))
n_tissues <- nrow(tissue_summary)

tissue_long <- tissue_summary %>%
  select(tissue_id, tissue_label, likely_log, confident_log) %>%
  pivot_longer(c(likely_log, confident_log), names_to = "type", values_to = "value")
tissue_long$type <- factor(tissue_long$type, levels = c("likely_log", "confident_log"),
                           labels = names(atlas_colours))

# radial labels, turned on the left half so that they are never upside down
label_r <- max(tissue_long$value, na.rm = TRUE) * 1.18
labels <- tissue_summary %>%
  transmute(tissue_id, tissue_label,
            angle_raw = 90 - (tissue_id - 0.5) * (360 / n_tissues),
            angle = ifelse(angle_raw < -90 | angle_raw > 90, angle_raw + 180, angle_raw),
            hjust = ifelse(angle_raw < -90 | angle_raw > 90, 1, 0),
            y = label_r)

p_radial <- ggplot() +
  geom_vline(data = data.frame(x = seq(0.5, n_tissues + 0.5, by = 1)), aes(xintercept = x),
             colour = "grey75", linewidth = 0.3) +
  geom_col(data = tissue_long, aes(x = tissue_id, y = value, fill = type), position = "identity", width = 0.85) +
  geom_text(data = labels, aes(x = tissue_id, y = y, label = tissue_label, angle = angle, hjust = hjust),
            size = 3.2, colour = "black") +
  scale_x_continuous(breaks = 1:n_tissues, limits = c(0.5, n_tissues + 0.5)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.55))) +
  scale_fill_manual(values = atlas_colours) +
  coord_polar(clip = "off") +
  theme_void(base_size = 14) +
  theme(legend.position = "none", plot.margin = margin(t = 10, r = 60, b = 10, l = 60, unit = "pt"))

sharing_long <- sharing %>%
  pivot_longer(c(likely, confident), names_to = "type", values_to = "value") %>%
  mutate(log_value = log10(value + 1))
sharing_long$type <- factor(sharing_long$type, levels = c("likely", "confident"), labels = names(atlas_colours))

p_sharing <- ggplot(sharing_long, aes(x = num_tissues, y = log_value, fill = type)) +
  geom_col(position = "identity", width = 0.8) +
  scale_fill_manual(values = atlas_colours) +
  scale_x_continuous(breaks = scales::breaks_pretty(n = 8)) +
  labs(x = "In number of tissues", y = expression(bold(Number ~ of ~ associations ~ (log[10])))) +
  theme_classic(base_size = 14) +
  theme(axis.text = element_text(colour = "black", size = 11),
        axis.title = element_text(colour = "black", size = 13, face = "bold"),
        axis.line = element_line(colour = "black", linewidth = 0.5),
        axis.ticks = element_line(colour = "black", linewidth = 0.5),
        legend.position = "bottom", legend.title = element_blank(), legend.text = element_text(size = 11),
        panel.grid.minor = element_blank())

save_panel(p_radial, "fig4_a_radial", 10, 10)
save_panel(p_sharing, "fig4_a_sharing", 10, 4)
