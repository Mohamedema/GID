# Appendix Figure S2b: Spearman correlation between the networks of the organ systems
source("R/setup.R")
library(ComplexHeatmap)

self_colour <- "grey88"

asim <- as.matrix(read.csv(file.path(results_dir, "anatomy_similarity.csv"), row.names = 1, check.names = FALSE))
stopifnot(identical(rownames(asim), colnames(asim)))
n <- nrow(asim)

# colour range: 95th percentile of the off diagonal correlations, rounded up to 0.05
range_lim  <- ceiling(quantile(abs(asim[row(asim) != col(asim)]), 0.95, na.rm = TRUE) * 20) / 20
legend_pts <- c(-range_lim, -range_lim / 2, 0, range_lim / 2, range_lim)
palette    <- colorRampPalette(c("#00688B", "#F0FFF0", "#228B22"))(200)

hc <- hclust(as.dist(1 - asim), method = "average")

# rows in tree order, columns in the reverse order, so that the upper left triangle is left
tri <- asim[hc$order, rev(hc$order)]
tri[row(tri) + col(tri) >= n + 1] <- NA
labels <- matrix(ifelse(is.na(tri), "", sprintf("%.2f", tri)), nrow = n, dimnames = dimnames(tri))

# back to the original row order, the tree puts the rows in order again
mat        <- tri[order(hc$order), ]
mat_labels <- labels[order(hc$order), ]
is_self    <- outer(rownames(mat), colnames(mat), "==")
is_hidden  <- is.na(mat) & !is_self

draw_organ_heatmap <- function() {
  ph <- ComplexHeatmap::pheatmap(mat, name = "Spearman\ncorrelation", color = palette, na_col = "white",
                                 breaks = seq(-range_lim, range_lim, length.out = 201),
                                 cluster_rows = hc, cluster_cols = FALSE,
                                 show_rownames = TRUE, show_colnames = TRUE,
                                 column_names_side = "top", angle_col = "45", row_names_side = "left",
                                 fontsize_row = 10, fontsize_col = 10, treeheight_row = 45,
                                 border_color = "white", display_numbers = mat_labels,
                                 number_color = "black", fontsize_number = 9,
                                 legend_breaks = legend_pts, legend_labels = sprintf("%.2f", legend_pts),
                                 main = "", fontfamily = "sans",
                                 cell_fun = function(j, i, x, y, width, height, fill) {
                                   if (is_self[i, j]) {
                                     grid::grid.rect(x, y, width, height, gp = grid::gpar(fill = self_colour, col = "white"))
                                   } else if (is_hidden[i, j]) {
                                     grid::grid.rect(x, y, width, height, gp = grid::gpar(fill = "white", col = "white"))
                                   }
                                 })
  print(ph)
}

save_drawn(draw_organ_heatmap, "supp2_b_organ_correlation", 8.5, 7.5)
