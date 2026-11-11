# Appendix Figure S8a: tissue specific complexes per breast cancer subtype
source("R/setup.R")
library(stringr)

subtypes <- read.csv(file.path(results_dir, "tcga", "Final_Rigorously_Tissue_Specific_Complexes.csv"))

bars <- subtypes %>%
  count(Tissue, name = "Count") %>%
  arrange(desc(Count)) %>%
  mutate(Tissue = factor(str_wrap(Tissue, width = 15), levels = str_wrap(Tissue, width = 15)),
         id = row_number())

# label angles, with a tilt of 10 degrees
tilt  <- 10
angle <- 90 - 360 * (bars$id - 0.5) / nrow(bars)
bars$hjust <- ifelse(angle < -90, 1, 0)
bars$angle <- ifelse(angle < -90, angle + 180, angle) + ifelse(bars$hjust == 0, tilt, -tilt)

p <- ggplot(bars, aes(x = as.factor(id), y = Count, fill = Count)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  ylim(-max(bars$Count) * 0.6, max(bars$Count) * 1.5) +
  coord_polar(start = 0) +
  geom_text(aes(x = id, y = Count + max(Count) * 0.1, label = Tissue, hjust = hjust),
            colour = "black", fontface = "bold", alpha = 0.9, size = 18, angle = bars$angle,
            inherit.aes = FALSE, lineheight = 0.8) +
  geom_text(aes(x = id, y = Count - max(Count) * 0.1, label = Count),
            colour = "white", fontface = "bold", size = 7, angle = bars$angle - ifelse(bars$hjust == 0, tilt, -tilt),
            inherit.aes = FALSE) +
  scale_fill_gradient(low = "#4682B4", high = "#B2182B") +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.title = element_blank(), panel.grid = element_blank(),
        plot.margin = unit(c(-1, -1, -1, -1), "cm"), legend.position = "none")

save_panel(p, "supp8_a_brca_subtypes", 10, 10)
