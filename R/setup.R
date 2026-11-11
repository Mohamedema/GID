suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

results_dir <- "results"
out_dir  <- "figures"
dir.create(out_dir, showWarnings = FALSE)

pt2mm <- function(pt) pt / .pt
lw    <- function(pt) pt / 0.75 / .pt

# ggplot panel: vector pdf and png
save_panel <- function(p, name, width, height, dpi = 600) {
  ggsave(file.path(out_dir, paste0(name, ".pdf")), p, width = width, height = height, device = cairo_pdf)
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = width, height = height, dpi = dpi, bg = "white")
}

# panels drawn with grid, ComplexHeatmap or circlize
save_drawn <- function(draw, name, width, height, dpi = 600) {
  cairo_pdf(file.path(out_dir, paste0(name, ".pdf")), width = width, height = height)
  draw()
  invisible(dev.off())
  png(file.path(out_dir, paste0(name, ".png")), width = width, height = height, units = "in",
      res = dpi, type = "cairo", bg = "white")
  draw()
  invisible(dev.off())
}
