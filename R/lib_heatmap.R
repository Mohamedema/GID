library(ComplexHeatmap)
library(circlize)
library(grid)

TOP_N      <- 3
Z_MAX      <- 5
LABEL_SIZE <- 14
CELL_W_CM  <- 0.70
CELL_H_CM  <- 0.85
FIG_W_IN   <- 20

z_colours <- colorRamp2(c(-Z_MAX, 0, Z_MAX), c("#2166AC", "white", "#B2182B"))

# row-wise z-score of the complex densities across `tissues`
density_z <- function(file, tissues) {
  m <- as.matrix(read.csv(file, row.names = 1, check.names = FALSE))
  m <- m[, intersect(tissues, colnames(m)), drop = FALSE]
  z <- (m - rowMeans(m, na.rm = TRUE)) / apply(m, 1, sd, na.rm = TRUE)
  z[!is.finite(z)] <- 0
  z
}

# Top complexes of each tissue by fold change. A complex listed in `manual` (matched on tissue and `key`)
# takes the place of the weakest pick of its tissue. Picks whose own z-score is not above 0 are dropped,
# except the manual ones.
pick_complexes <- function(sel, z, manual, key, col_order) {
  picks <- sel %>%
    group_by(Tissue) %>%
    arrange(desc(Fold_Change), .by_group = TRUE) %>%
    slice_head(n = TOP_N) %>%
    ungroup()

  targets <- inner_join(sel, manual, by = c("Tissue", key))
  for (i in seq_len(nrow(targets))) {
    tis <- targets$Tissue[i]
    id  <- targets$Complex_ID_in_Model[i]
    if (any(picks$Tissue == tis & picks$Complex_ID_in_Model == id)) next
    weakest <- picks %>% filter(Tissue == tis) %>% arrange(Fold_Change) %>% slice_head(n = 1)
    if (nrow(weakest) == 0) next
    picks <- picks %>%
      filter(!(Tissue == tis & Complex_ID_in_Model == weakest$Complex_ID_in_Model)) %>%
      bind_rows(targets[i, ])
  }

  picks$own_z <- mapply(function(id, tis)
    if (id %in% rownames(z) && tis %in% colnames(z)) z[id, tis] else NA_real_,
    picks$Complex_ID_in_Model, picks$Tissue)
  is_manual <- picks[[key]] %in% manual[[key]]
  picks <- picks[(!is.na(picks$own_z) & picks$own_z > 0) | is_manual, ]

  picks %>%
    mutate(rank = match(Tissue, col_order)) %>%
    arrange(rank, desc(Fold_Change)) %>%
    group_by(Tissue, complex_name) %>%
    slice(1) %>%
    ungroup() %>%
    arrange(rank, desc(Fold_Change))
}

# complexes in rows, tissues in columns, a diamond in the tissue each complex was assigned to
complex_heatmap <- function(picks, z, col_order) {
  stopifnot("complexes missing from the density matrix" = all(picks$Complex_ID_in_Model %in% rownames(z)))
  mat  <- z[picks$Complex_ID_in_Model, col_order, drop = FALSE]
  home <- match(picks$Tissue, colnames(mat))

  ht <- Heatmap(mat, col = z_colours, na_col = "grey85", show_heatmap_legend = FALSE,
                cluster_rows = FALSE, cluster_columns = FALSE,
                row_split = factor(picks$Tissue, levels = col_order[col_order %in% picks$Tissue]),
                row_gap = unit(0.6, "mm"),
                row_title_gp = gpar(fontsize = LABEL_SIZE, fontface = "bold", col = "black"), row_title_rot = 0,
                row_labels = picks$complex_name,
                row_names_gp = gpar(fontsize = LABEL_SIZE, fontface = "italic", col = "black"),
                column_names_gp = gpar(fontsize = LABEL_SIZE, fontface = "bold", col = "black"),
                column_names_rot = 45, column_title = NULL,
                cell_fun = function(j, i, x, y, width, height, fill) {
                  if (!is.na(home[i]) && j == home[i])
                    grid.points(x, y, pch = 23, size = unit(2.4, "mm"), gp = gpar(col = "black", fill = "black", lwd = 0.8))
                },
                rect_gp = gpar(col = "grey90", lwd = 0.4),
                border = TRUE, border_gp = gpar(col = "black", lwd = 1.2),
                width = unit(ncol(mat) * CELL_W_CM, "cm"), height = unit(nrow(mat) * CELL_H_CM, "cm"))

  list(draw = function() draw(ht, background = "white", padding = unit(c(8, 4, 8, 4), "mm")),
       height = nrow(mat) * CELL_H_CM / 2.54 + 3.5)
}

# colour key and diamond key
draw_heatmap_legend <- function() {
  key_z <- Legend(col_fun = z_colours, title = "Density z-score",
                  title_gp = gpar(fontsize = 10, fontface = "bold", col = "black"),
                  at = c(-Z_MAX, -Z_MAX / 2, 0, Z_MAX / 2, Z_MAX),
                  labels = c(paste0("\u2264 -", Z_MAX), as.character(-Z_MAX / 2), "0",
                             as.character(Z_MAX / 2), paste0("\u2265 ", Z_MAX)),
                  labels_gp = gpar(fontsize = 9, col = "black"),
                  legend_height = unit(3.5, "cm"), grid_width = unit(0.45, "cm"), direction = "vertical")
  key_home <- Legend(labels = "Home tissue (Fold-Change ranked, sanity-filtered)", type = "points", pch = 23,
                     legend_gp = gpar(col = "black", fill = "black", lwd = 1), background = "white",
                     labels_gp = gpar(fontsize = 9, col = "black"), size = unit(3, "mm"))
  keys <- packLegend(key_z, key_home, direction = "vertical", gap = unit(4, "mm"))

  grid.newpage()
  grid.rect(gp = gpar(fill = "white", col = NA))
  pushViewport(viewport(x = 0.5, y = 0.5, just = c("centre", "centre")))
  draw(keys)
  popViewport()
}
