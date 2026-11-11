FIG_W <- 7.2          # printed width of Figure 2 (inches)
ROW_H <- 3.8
FIG_H <- 2 * ROW_H
COL_L <- FIG_W / 2
COL_R <- FIG_W - COL_L

FS_TAG   <- 10
FS_TITLE <- 7.5
FS_TEXT  <- 7

omic_colours <- c("Transcriptomics" = "#E64B35", "Proteomics" = "#F39B7F", "CRISPR-Cas9" = "#4DBBD5",
                  "Methylation" = "#00A087", "CNV" = "#3C5488", "Multi-Omics" = "#B09C85")
omic_order <- names(omic_colours)

theme_nature <- function(base_size = FS_TITLE) {
  theme_bw(base_size = base_size) +
    theme(panel.grid.major  = element_line(colour = "grey92", linewidth = lw(0.4)),
          panel.grid.minor  = element_blank(),
          panel.border      = element_rect(colour = "black", fill = NA, linewidth = lw(0.7)),
          panel.background  = element_rect(fill = "white"),
          axis.title        = element_text(size = FS_TITLE, face = "bold"),
          axis.text         = element_text(size = FS_TEXT, colour = "black"),
          axis.ticks        = element_line(colour = "black", linewidth = lw(0.5)),
          axis.ticks.length = unit(0.09, "cm"),
          legend.background = element_rect(fill = "white", colour = "grey80", linewidth = lw(0.4)),
          legend.key        = element_rect(fill = "white"),
          legend.key.size   = unit(0.30, "cm"),
          legend.text       = element_text(size = FS_TEXT),
          legend.title      = element_text(size = FS_TITLE, face = "bold"),
          legend.spacing.y  = unit(0.02, "cm"),
          plot.margin       = margin(10, 5, 2, 2),
          plot.tag          = element_text(size = FS_TAG, face = "bold", hjust = 0, vjust = 1),
          plot.tag.position = c(0, 1))
}

# legend inside the plot, for ggplot2 3.4 and for 3.5 or newer
legend_inside <- function(pos, just) {
  if (utils::packageVersion("ggplot2") >= "3.5.0") {
    theme(legend.position = "inside", legend.position.inside = pos, legend.justification = just)
  } else {
    theme(legend.position = pos, legend.justification = just)
  }
}

trapezoid_auc <- function(fpr, tpr) {
  o <- order(fpr)
  x <- fpr[o]
  y <- tpr[o]
  sum(diff(x) * (head(y, -1) + tail(y, -1))) / 2
}

# ROC curves of the five omic layers and of the multi-omics model, for the reference set `db` ("corum" or "string")
roc_plot <- function(db) {
  prefix <- c("Transcriptomics" = "mrna", "Proteomics" = "protein", "CRISPR-Cas9" = "crispr",
              "Methylation" = "methylation", "CNV" = "cnv", "Multi-Omics" = "all_omics")
  roc <- bind_rows(lapply(names(prefix), function(o) {
    d <- read.csv(file.path(results_dir, paste0(prefix[[o]], "_roc_", db, ".csv")))
    colnames(d) <- c("FPR", "TPR")
    d$omic <- o
    d$auc  <- trapezoid_auc(d$FPR, d$TPR)
    d
  }))
  auc_table <- roc %>%
    group_by(omic) %>%
    summarise(auc = first(auc), .groups = "drop") %>%
    arrange(desc(auc))
  roc$omic <- factor(roc$omic, levels = auc_table$omic)
  legend_labels <- setNames(sprintf("%s  (AUC = %.4f)", auc_table$omic, auc_table$auc), auc_table$omic)

  ggplot(roc, aes(x = FPR, y = TPR, colour = omic)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black", linewidth = lw(0.6)) +
    geom_line(linewidth = lw(1.1), alpha = 0.92) +
    scale_x_continuous(name = "False Positive Rate", limits = c(0, 1), breaks = seq(0, 1, 0.2),
                       expand = expansion(mult = 0.005)) +
    scale_y_continuous(name = "True Positive Rate", limits = c(0, 1), breaks = seq(0, 1, 0.2),
                       expand = expansion(mult = 0.005)) +
    scale_colour_manual(values = omic_colours[levels(roc$omic)], labels = legend_labels) +
    guides(colour = guide_legend(override.aes = list(linewidth = lw(1.8)), byrow = TRUE)) +
    theme_nature() +
    legend_inside(c(0.99, 0.02), c("right", "bottom")) +
    theme(legend.margin = margin(3, 4, 3, 4),
          legend.key.height = unit(0.27, "cm"),
          legend.key.width  = unit(0.35, "cm"))
}
