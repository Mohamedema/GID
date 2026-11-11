# Figure 6c: terms enriched among the genes of the network, blood and immune related terms only
source("R/setup.R")
library(stringr)

gp <- read.csv(file.path(results_dir, "gprofiler_first_degree.csv"))
colnames(gp)[colnames(gp) == "adjusted_p_value"] <- "p_value"
n_query <- gp$query_size[1]

keywords <- c("leuk", "lymph", "blood", "hemat", "haemat", "myeloid", "erythro", "platelet", "marrow", "thrombo",
              "granulocyte", "neutrophil", "monocyte", "spleen", "thymus", "tonsil", "b cell", "t cell", "b-cell",
              "t-cell", "immune", "immunolog", "cytopen")
gp$relevant <- str_detect(str_to_lower(gp$term_name), paste(keywords, collapse = "|"))

theme_enrich <- function(base_size = 10) {
  theme_classic(base_size = base_size) +
    theme(axis.line = element_line(colour = "black", linewidth = 0.5),
          axis.text = element_text(colour = "black", size = base_size - 1),
          axis.text.y = element_text(size = 8.5, hjust = 1),
          axis.title.x = element_text(size = base_size, margin = margin(t = 6)),
          axis.title.y = element_blank(),
          panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.35),
          panel.grid.major.y = element_blank(),
          panel.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white", colour = NA),
          legend.position = "bottom", legend.box = "horizontal",
          legend.key = element_rect(fill = "white"),
          legend.text = element_text(size = base_size - 2),
          legend.title = element_text(size = base_size - 2),
          legend.spacing.x = unit(0.3, "cm"),
          plot.title = element_text(size = base_size + 1, face = "bold", hjust = 0.5, margin = margin(b = 10)),
          plot.margin = margin(12, 16, 8, 10))
}

enrichment_dots <- function(source_name, title, low, high, wrap, n_terms = 10, min_overlap = 3) {
  d <- gp %>%
    filter(source == source_name, relevant, !is.na(p_value), p_value < 0.05, intersection_size >= min_overlap) %>%
    arrange(p_value) %>%
    head(n_terms) %>%
    mutate(term = str_wrap(term_name, width = wrap),
           term = factor(term, levels = rev(unique(term))),
           ratio = intersection_size / n_query,
           log_p = -log10(p_value))

  size_range  <- range(d$intersection_size)
  size_breaks <- pretty(size_range, n = 3)
  size_breaks <- size_breaks[size_breaks >= size_range[1] & size_breaks <= size_range[2]]
  if (length(size_breaks) < 2) size_breaks <- size_range
  ratio_breaks <- round(pretty(range(d$ratio), n = 3), 3)
  ratio_breaks <- ratio_breaks[ratio_breaks >= min(d$ratio) & ratio_breaks <= max(d$ratio)]

  ggplot(d, aes(x = log_p, y = term, size = intersection_size, colour = ratio)) +
    geom_point(alpha = 0.9) +
    scale_size_continuous(name = "Overlap genes count", range = c(3, 12),
                          breaks = size_breaks, labels = as.character(size_breaks),
                          guide = guide_legend(title.position = "bottom", title.hjust = 0.5, label.position = "bottom",
                                               override.aes = list(colour = "grey60", alpha = 0.6), order = 1)) +
    scale_colour_gradient(name = "Enrichment\nratio", low = low, high = high,
                          breaks = ratio_breaks, labels = sprintf("%.2f", ratio_breaks),
                          guide = guide_colorbar(title.position = "bottom", title.hjust = 0.5,
                                                 barwidth = unit(3.5, "cm"), barheight = unit(0.5, "cm"), order = 2)) +
    scale_x_continuous(name = expression(-log[10](Adjusted~italic(P)-value)), expand = expansion(mult = c(0.02, 0.12))) +
    labs(title = title) +
    theme_enrich()
}

go_bp <- enrichment_dots("GO:BP", "GO Biological Process", "#C7E9C0", "#00441B", wrap = 34)
hp    <- enrichment_dots("HP", "Human Phenotype: Clinical Abnormalities", "#C6DBEF", "#084594", wrap = 32)

save_panel(go_bp, "fig6_c_go_bp", 8, 6.5)
save_panel(hp, "fig6_c_hp", 8, 6.5)
