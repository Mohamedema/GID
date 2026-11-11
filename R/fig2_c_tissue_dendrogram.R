# Figure 2c: circular dendrogram of the tissue networks, ring coloured by organ system
source("R/setup.R")
source("R/lib_fig2.R")
library(grid)

tsim  <- read.csv(file.path(results_dir, "tissue_similarity.csv"), row.names = 1, check.names = FALSE)
tmeta <- read.csv("data/tissue_anatomy.csv")
tmeta <- tmeta[match(rownames(tsim), tmeta$Tissue), ]

npg <- c("#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2",
         "#DC0000", "#7E6148", "#B09C85", "#9A709E", "#5A9599", "#C16622", "#8996A6")
systems         <- unique(tmeta$Anatomy)
anatomy_colours <- setNames(npg[seq_along(systems)], systems)

phy         <- ape::as.phylo(hclust(dist(as.matrix(tsim)), method = "ward.D2"))
tips        <- phy$tip.label
tip_anatomy <- tmeta$Anatomy[match(tips, tmeta$Tissue)]
n_tips      <- length(tips)

# fan layout of ape: tip k at angle 2 pi (k - 1) / n, inner nodes at the mean angle of their first and last child
fan_layout <- function(phy) {
  phy   <- ape::reorder.phylo(phy, "cladewise")
  n     <- ape::Ntip(phy)
  edge  <- phy$edge
  depth <- ape::node.depth.edgelength(phy)
  pos   <- rep(NA_real_, n + phy$Nnode)
  pos[edge[edge[, 2] <= n, 2]] <- seq_len(n)
  kid1 <- kid2 <- rep(NA_integer_, n + phy$Nnode)
  settle <- function(node) {
    if (node <= n) return(invisible(NULL))
    kids <- edge[edge[, 1] == node, 2]
    for (k in kids) settle(k)
    kid1[node] <<- kids[1]
    kid2[node] <<- kids[length(kids)]
    pos[node]  <<- (pos[kids[1]] + pos[kids[length(kids)]]) / 2
  }
  settle(n + 1)
  list(n = n, edge = edge, theta = 2 * pi * (pos - 1) / n,
       rel_r = depth / max(depth[1:n]), kid1 = kid1, kid2 = kid2)
}
lay <- fan_layout(phy)

text_width <- function(txt, size, face = "plain") {
  pdf(NULL)
  on.exit(dev.off())
  vapply(txt, function(s)
    convertWidth(grobWidth(textGrob(s, gp = gpar(fontsize = size, fontface = face, fontfamily = "sans"))),
                 "in", valueOnly = TRUE), numeric(1), USE.NAMES = FALSE)
}

text_corners <- function(x, y, w, h, ang, hj) {
  a  <- ang * pi / 180
  u  <- c(cos(a), sin(a))
  nv <- c(-sin(a), cos(a))
  s  <- c(x, y) - hj * w * u
  e  <- s + w * u
  rbind(s + h / 2 * nv, s - h / 2 * nv, e + h / 2 * nv, e - h / 2 * nv)
}

# sizes in inches
RING_GAP   <- 0.03
RING_THICK <- 0.065
LABEL_GAP  <- 0.04
RIB_GAP    <- 0.10
RIB_H      <- 0.10
RIB_ANGLE  <- 50
RIB_WIDTH  <- 1.7     # ribbon width in ring diameters

used_sys <- systems[systems %in% tip_anatomy]
n_sys    <- length(used_sys)
w_tip    <- text_width(tips, FS_TEXT, "italic")
w_sys    <- text_width(used_sys, FS_TEXT)
h_text   <- FS_TEXT / 72 * 0.9

tree_geometry <- function(R_tip) {
  ring_in  <- R_tip + RING_GAP
  ring_out <- ring_in + RING_THICK
  r_lab    <- ring_out + LABEL_GAP
  th       <- lay$theta[1:n_tips]
  right    <- cos(th) >= -1e-9
  lab <- data.frame(label = tips, x = r_lab * cos(th), y = r_lab * sin(th),
                    angle = ifelse(right, th * 180 / pi, th * 180 / pi - 180),
                    hjust = ifelse(right, 0, 1))
  corners <- do.call(rbind, lapply(seq_len(n_tips), function(i)
    text_corners(lab$x[i], lab$y[i], w_tip[i], h_text, lab$angle[i], lab$hjust[i])))
  rib_w   <- RIB_WIDTH * 2 * ring_out
  box_w   <- rib_w / n_sys
  rib_top <- min(corners[, 2]) - RIB_GAP
  rib_bot <- rib_top - RIB_H
  xc      <- -rib_w / 2 + (seq_len(n_sys) - 0.5) * box_w
  sys_corners <- do.call(rbind, lapply(seq_len(n_sys), function(i)
    text_corners(xc[i], rib_bot - 0.03, w_sys[i], h_text, RIB_ANGLE, 1)))
  list(R_tip = R_tip, ring_in = ring_in, ring_out = ring_out, lab = lab,
       rib_w = rib_w, box_w = box_w, rib_top = rib_top, rib_bot = rib_bot, xc = xc,
       xr = range(c(corners[, 1], sys_corners[, 1], -rib_w / 2, rib_w / 2)),
       yr = range(c(corners[, 2], sys_corners[, 2], rib_top, rib_bot)))
}

# largest tree radius for which labels and ribbon still fit in the cell of the panel
C_W <- COL_L
C_H <- ROW_H
PAD <- 0.04
geo <- NULL
for (R in seq(1.6, 0.25, by = -0.005)) {
  g <- tree_geometry(R)
  if (diff(g$xr) <= C_W - 2 * PAD && diff(g$yr) <= C_H - 2 * PAD) { geo <- g; break }
}
if (is.null(geo)) stop("panel c does not fit: increase COL_L / ROW_H or reduce FS_TEXT")

th     <- lay$theta
rr     <- lay$rel_r * geo$R_tip
edge   <- lay$edge
radial <- data.frame(x = rr[edge[, 1]] * cos(th[edge[, 2]]), y = rr[edge[, 1]] * sin(th[edge[, 2]]),
                     xend = rr[edge[, 2]] * cos(th[edge[, 2]]), yend = rr[edge[, 2]] * sin(th[edge[, 2]]))
nodes  <- (n_tips + 1):length(th)
nodes  <- nodes[rr[nodes] > 1e-6]
arcs <- do.call(rbind, lapply(nodes, function(nd) {
  a <- seq(th[lay$kid1[nd]], th[lay$kid2[nd]],
           length.out = max(3, ceiling(abs(th[lay$kid2[nd]] - th[lay$kid1[nd]]) / (pi / 90))))
  data.frame(node = nd, x = rr[nd] * cos(a), y = rr[nd] * sin(a))
}))

half_width <- (2 * pi / n_tips) / 2 * 0.85
ring <- do.call(rbind, lapply(seq_len(n_tips), function(i) {
  a <- seq(th[i] - half_width, th[i] + half_width, length.out = 12)
  data.frame(id = i, anatomy = tip_anatomy[i],
             x = c(geo$ring_in * cos(a), rev(geo$ring_out * cos(a))),
             y = c(geo$ring_in * sin(a), rev(geo$ring_out * sin(a))))
}))

rib <- data.frame(sys = used_sys, xmin = geo$xc - geo$box_w / 2, xmax = geo$xc + geo$box_w / 2,
                  ymin = geo$rib_bot, ymax = geo$rib_top)
rib_lab <- data.frame(sys = used_sys, x = geo$xc, y = geo$rib_bot - 0.03)

cx <- mean(geo$xr)
cy <- mean(geo$yr)

p_c <- ggplot() +
  geom_segment(data = radial, aes(x = x, y = y, xend = xend, yend = yend),
               colour = "grey40", linewidth = lw(0.5), lineend = "butt") +
  geom_path(data = arcs, aes(x = x, y = y, group = node), colour = "grey40", linewidth = lw(0.5)) +
  geom_polygon(data = ring, aes(x = x, y = y, group = id, fill = anatomy), colour = NA) +
  geom_text(data = geo$lab, aes(x = x, y = y, label = label, angle = angle, hjust = hjust),
            vjust = 0.5, size = pt2mm(FS_TEXT), fontface = "italic", colour = "black") +
  geom_rect(data = rib, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = sys), colour = NA) +
  annotate("rect", xmin = min(rib$xmin), xmax = max(rib$xmax), ymin = geo$rib_bot, ymax = geo$rib_top,
           fill = NA, colour = "black", linewidth = lw(0.5)) +
  geom_text(data = rib_lab, aes(x = x, y = y, label = sys), angle = RIB_ANGLE, hjust = 1, vjust = 0.5,
            size = pt2mm(FS_TEXT), colour = "black") +
  scale_fill_manual(values = anatomy_colours, guide = "none") +
  coord_fixed(xlim = cx + c(-C_W / 2, C_W / 2), ylim = cy + c(-C_H / 2, C_H / 2), expand = FALSE, clip = "off") +
  labs(tag = "c") +
  theme_void() +
  theme(plot.margin = margin(0, 0, 0, 0),
        plot.tag = element_text(size = FS_TAG, face = "bold", hjust = 0, vjust = 1),
        plot.tag.position = c(0, 1))

save_panel(p_c, "fig2_c_tissue_dendrogram", C_W, C_H)
