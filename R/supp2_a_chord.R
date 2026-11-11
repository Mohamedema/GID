# Appendix Figure S2a: chord diagram of the tissues and their organ systems
source("R/setup.R")
library(circlize)

anatomy_map <- c(
  "Adrenal Gland" = "Endocrine", "Biliary Tract" = "Digestive", "Bladder" = "Urinary",
  "Bone" = "Skeletal", "Breast" = "Reproductive", "Central Nervous System" = "Neural",
  "Cervix" = "Reproductive", "Endometrium" = "Reproductive", "Esophagus" = "Digestive",
  "Haematopoietic and Lymphoid" = "Immune", "Head and Neck" = "Head and Neck",
  "Kidney" = "Urinary", "Large Intestine" = "Digestive", "Liver" = "Digestive",
  "Lung" = "Respiratory", "Ovary" = "Reproductive", "Pancreas" = "Digestive",
  "Peripheral Nervous System" = "Neural", "Placenta" = "Reproductive",
  "Prostate" = "Reproductive", "Skin" = "Skin", "Soft Tissue" = "Soft Tissue",
  "Stomach" = "Digestive", "Testis" = "Reproductive", "Thyroid" = "Endocrine",
  "Uterus" = "Reproductive", "Vulva" = "Reproductive", "Small Intestine" = "Digestive"
)
links <- data.frame(tissue = names(anatomy_map), organ = unname(anatomy_map), value = 1,
                    stringsAsFactors = FALSE)

# organ systems in one alphabetical block, tissues in another
organ_order  <- sort(unique(links$organ))
tissue_order <- sort(unique(links$tissue))
sector_order <- c(organ_order, tissue_order)
block <- factor(setNames(c(rep("Organs", length(organ_order)), rep("Tissues", length(tissue_order))), sector_order),
                levels = c("Organs", "Tissues"))

organ_palette <- c("Digestive" = "#E64B35", "Endocrine" = "#4DBBD5", "Head and Neck" = "#00A087",
                   "Immune" = "#3C5488", "Neural" = "#F39B7F", "Reproductive" = "#8491B4",
                   "Respiratory" = "#91D1C2", "Skeletal" = "#DC0000", "Skin" = "#7E6148",
                   "Soft Tissue" = "#B09C85", "Urinary" = "#374E55")
sector_colours <- c(organ_palette[organ_order],
                    setNames(organ_palette[links$organ[match(tissue_order, links$tissue)]], tissue_order))
link_colours   <- adjustcolor(organ_palette[links$organ], alpha.f = 0.55)

draw_chord <- function() {
  circos.clear()
  circos.par(start.degree = 90, track.margin = c(0.005, 0.005), points.overflow.warning = FALSE)
  chordDiagram(links[, c("tissue", "organ", "value")], order = sector_order, group = block,
               big.gap = 8, small.gap = 1, grid.col = sector_colours, col = link_colours, transparency = 0,
               directional = 1, direction.type = c("arrows", "diffHeight"), link.arr.type = "big.arrow",
               diffHeight = -0.02, annotationTrack = "grid", preAllocateTracks = list(track.height = mm_h(4)),
               link.sort = TRUE, link.decreasing = TRUE)
  circos.trackPlotRegion(track.index = 1, bg.border = NA, panel.fun = function(x, y) {
    circos.text(mean(get.cell.meta.data("xlim")), get.cell.meta.data("ylim")[1] + mm_y(2),
                get.cell.meta.data("sector.index"), facing = "clockwise", niceFacing = TRUE,
                cex = 2, adj = c(0, 0.5), col = "grey15")
  })
  circos.clear()
}

save_drawn(draw_chord, "supp2_a_chord", 40, 36, dpi = 300)
