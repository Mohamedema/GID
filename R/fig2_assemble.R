# Figure 2: panels a to d on one page at the printed size.
# The plot areas of a, b and d are squares of the same size, and the axis line of d
# lies on the left edge of the frame of b.
for (f in c("fig2_a_roc_corum.R", "fig2_b_clustering_space.R",
            "fig2_c_tissue_dendrogram.R", "fig2_d_jaccard.R"))
  source(file.path("R", f))

library(grid)

# space (inches) around the plot area of a plot
margins_in <- function(g) {
  p <- g$layout[g$layout$name == "panel", ]
  w <- sapply(seq_along(g$widths),  function(i) convertWidth(g$widths[i],   "in", valueOnly = TRUE))
  h <- sapply(seq_along(g$heights), function(i) convertHeight(g$heights[i], "in", valueOnly = TRUE))
  c(left = sum(w[seq_len(p$l - 1)]), right = sum(w[-seq_len(p$r)]),
    top = sum(h[seq_len(p$t - 1)]), bottom = sum(h[-seq_len(p$b)]))
}

# draw g so that its plot area has its left edge at x, its top edge at y_top, and size W x H
place_plot <- function(g, m, x, y_top, W, H) {
  pushViewport(viewport(x = unit(x - m[["left"]], "in"), y = unit(y_top - H - m[["bottom"]], "in"),
                        width = unit(W + m[["left"]] + m[["right"]], "in"),
                        height = unit(H + m[["top"]] + m[["bottom"]], "in"),
                        just = c("left", "bottom")))
  grid.draw(g)
  popViewport()
}

draw_fig2 <- function() {
  grid.newpage()
  ga <- ggplotGrob(p_a)
  gb <- ggplotGrob(p_b)
  gd <- ggplotGrob(p_d)
  ma <- margins_in(ga)
  mb <- margins_in(gb)
  md <- margins_in(gd)
  top <- max(ma[["top"]], mb[["top"]], md[["top"]])
  bot <- max(ma[["bottom"]], mb[["bottom"]], md[["bottom"]])
  S <- min(COL_L - ma[["left"]] - ma[["right"]], COL_R - mb[["left"]] - mb[["right"]], ROW_H - top - bot)
  x_a <- ma[["left"]]
  x_b <- COL_L + mb[["left"]]
  x_d <- x_b
  tag_d <- (COL_L - (x_d - md[["left"]])) / (S + md[["left"]] + md[["right"]])
  gd <- ggplotGrob(p_d + theme(plot.tag.position = c(tag_d, 1), plot.background = element_blank()))
  place_plot(ga, ma, x_a, FIG_H - top, S, S)
  place_plot(gb, mb, x_b, FIG_H - top, S, S)
  place_plot(gd, md, x_d, ROW_H - top, S, S)
  print(p_c + theme(plot.background = element_blank()),
        vp = viewport(x = unit(0, "in"), y = unit(0, "in"), width = unit(COL_L, "in"),
                      height = unit(ROW_H, "in"), just = c("left", "bottom")))
}

save_drawn(draw_fig2, "fig2_abcd", FIG_W, FIG_H)
