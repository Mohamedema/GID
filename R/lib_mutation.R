# Panels of the mutation analysis (Figure 5, Appendix Figures S6 and S7).
# The Python analysis writes one set of tables per gene (<GENE>_edges.csv.gz, <GENE>_tissue_table.csv, ...);
# load_gene() reads them and every panel_*() function draws one panel from the result.
library(ggplot2)
library(grid)
library(ggrepel)

FDR_CUT   <- 0.05
N_LABELS  <- 6           # volcano: labelled partners per side
N_HEATMAP <- 20          # heatmap rows
N_NETWORK <- 20          # network partners
N_TISSUES <- 12          # heatmap columns, the tissues with most mutated lines
LAB_UP    <- "Gained"    # Delta > 0
LAB_DOWN  <- "Lost"      # Delta < 0

FONT       <- "Arial"
RASTER_DPI <- 600
LINE_PT    <- 0.5        # axis and tick line weight in points

COL <- list(up = "#D2553B", down = "#1F78A6", ns = "#D4D4D4", mid = "#F4F4F4", bar = "#6E6E6E", bar_pale = "#C4C4C4",
            both = "#2A78D6", only_without = "#EB6834", only_with = "#1BAF7A",
            degree = c("1" = "#7D0462", "2" = "#A53E87", "3" = "#C670AA", "4+" = "#E39ECB"))
dir_cols <- c(down = COL$down, up = COL$up, ns = COL$ns)

# Position of the panels on the page in mm (left, top, width, height) and the letter each carries.
# The names are the panels of the analysis: a tissues, b volcano, c cell-line scores, d heatmap, e network,
# f degrees, g with and without tissue, h heaviest tissue left out, i permutations.
FIG5 <- list(a = list(letter = "a", box = c(0, 0, 46, 60)),
             b = list(letter = "b", box = c(46, 0, 67, 60)),
             c = list(letter = "c", box = c(113, 0, 67, 60)),
             e = list(letter = "d", box = c(0, 60, 90, 70)),
             f = list(letter = "e", box = c(90, 60, 90, 70)),
             g = list(letter = "f", box = c(0, 130, 60, 54)),
             h = list(letter = "g", box = c(60, 130, 60, 54)),
             i = list(letter = "h", box = c(120, 130, 60, 54)))
SUPP6 <- list(d = list(letter = NULL, box = c(0, 60, 108, 70)))
SUPP7 <- list(a = list(letter = "a", box = c(0, 0, 46, 60)),
              b = list(letter = "b", box = c(46, 0, 67, 60)),
              c = list(letter = "c", box = c(113, 0, 67, 60)))

UTF8  <- isTRUE(l10n_info()[["UTF-8"]])        # the Greek delta and the minus sign need a UTF-8 session
LW    <- LINE_PT / 0.75 / ggplot2::.pt         # points to ggplot linewidth
PT    <- ggplot2::.pt
DELTA <- if (UTF8) "\u0394" else expression(Delta)
delta_title <- function(rest) if (UTF8) paste0("\u0394 ", rest) else bquote(Delta ~ .(rest))
Y_VOLCANO   <- if (UTF8) expression("\u2212log"[10] * "(FDR)") else expression("-log"[10] * "(FDR)")

# helpers
fmt       <- function(n) format(n, big.mark = ",", trim = TRUE, scientific = FALSE)
fmt_short <- function(n) ifelse(n >= 1e4, paste0(round(n / 1000), "k"), fmt(n))
wrap_text <- function(s, width) paste(strwrap(s, width = width), collapse = "\n")

minus <- function(x) {                         # axis numbers with a true minus sign
  out <- format(x, trim = TRUE, drop0trailing = TRUE)
  out[is.na(x)] <- NA
  if (UTF8) sub("-", "\u2212", out, fixed = TRUE) else out
}

dense_points <- function(...) {                # large point clouds are rasterised inside the pdf when ggrastr is installed
  layer <- geom_point(...)
  if (requireNamespace("ggrastr", quietly = TRUE))
    ggrastr::rasterise(layer, dpi = RASTER_DPI, dev = if (requireNamespace("ragg", quietly = TRUE)) "ragg" else "cairo")
  else layer
}

sci <- function(p) {                           # plotmath text for a small number
  if (is.na(p)) return('"NA"')
  if (p <= 0) return('"< 1e-300"')
  if (p >= 0.001) return(sprintf('"%.3f"', p))
  e <- floor(log10(p)); m <- round(p / 10^e)
  if (m == 10) { m <- 1; e <- e + 1 }
  sprintf("%d %%*%% 10^{-%d}", m, -e)
}

wilson <- function(k, n, z = 1.959964) {       # 95% interval of a proportion
  if (n == 0) return(c(NA_real_, NA_real_))
  p <- k / n; den <- 1 + z^2 / n
  mid  <- (p + z^2 / (2 * n)) / den
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / den
  c(max(0, mid - half), min(1, mid + half))
}

bezier <- function(x0, y0, x1, y1, pull = 0.3, n = 40) {   # arc between two nodes, bent towards the centre
  t  <- seq(0, 1, length.out = n)
  cx <- pull * (x0 + x1) / 2; cy <- pull * (y0 + y1) / 2
  data.frame(x = (1 - t)^2 * x0 + 2 * (1 - t) * t * cx + t^2 * x1,
             y = (1 - t)^2 * y0 + 2 * (1 - t) * t * cy + t^2 * y1)
}

theme_fig <- function() {
  theme_classic(base_size = 7, base_family = FONT) +
    theme(axis.line         = element_line(linewidth = LW, colour = "black"),
          axis.ticks        = element_line(linewidth = LW, colour = "black"),
          axis.ticks.length = unit(1, "mm"),
          axis.text         = element_text(size = 6, colour = "black"),
          axis.title        = element_text(size = 7, colour = "black"),
          strip.background  = element_blank(),
          strip.text        = element_text(size = 6.5, colour = "black", margin = margin(b = 1, unit = "mm")),
          legend.position      = "top",
          legend.justification = "left",
          legend.direction     = "vertical",
          legend.title         = element_blank(),
          legend.text          = element_text(size = 6, colour = "black"),
          legend.key.size      = unit(2.7, "mm"),
          legend.margin        = margin(0, 0, 0, 0),
          legend.box.margin    = margin(0, 0, 0, 0),
          legend.box.spacing   = unit(0.8, "mm"),
          legend.background    = element_blank(),
          legend.key           = element_blank(),
          plot.margin          = margin(5.2, 3, 1.5, 2.5, unit = "mm"))
}

empty_panel <- function(msg) {
  ggplot() + annotate("text", x = 0, y = 0, label = msg, size = 6 / PT, family = FONT, colour = "grey35") +
    theme_void(base_family = FONT)
}

repel_labels <- function(lab, x_abs) {         # labels in two columns on the empty inner side of a volcano
  left   <- lab$dir == "down"
  target <- ifelse(left, -1, 1) * 0.45 * x_abs
  geom_text_repel(data = lab, aes(x = Delta, y = y, label = label), inherit.aes = FALSE, size = 6 / PT,
                  family = FONT, colour = "black", bg.color = "white", bg.r = 0.12,
                  nudge_x = target - lab$Delta, hjust = ifelse(left, 0, 1), direction = "y",
                  segment.size = 0.2, segment.colour = "grey30", min.segment.length = 0, box.padding = 0.12,
                  point.padding = 0.1, max.overlaps = Inf, max.iter = 30000, max.time = 3, seed = 1,
                  show.legend = FALSE)
}

top_per_side <- function(x, n) {
  x <- x[order(x$q, -abs(x$Std_Effect)), ]
  rbind(utils::head(x[x$dir == "down", ], n), utils::head(x[x$dir == "up", ], n))
}

read_table <- function(gene, suffix) {
  path <- file.path(results_dir, "mutations", gene, paste0(gene, suffix))
  if (!file.exists(path)) stop("File not found: ", path, call. = FALSE)
  tryCatch(utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE), error = function(e) data.frame())
}

# all tables of one gene
load_gene <- function(gene) {
  edges <- read_table(gene, "_edges.csv.gz")
  smry  <- read_table(gene, "_summary.csv")
  edges$q   <- edges$FDR
  edges$y   <- -log10(pmax(edges$q, 1e-300))
  edges$alt <- edges$q < FDR_CUT
  edges$dir <- ifelse(edges$Delta > 0, "up", "down")
  first <- edges[edges$Degree == 1, ]                       # direct partners of the gene
  first <- first[order(first$q, -abs(first$Std_Effect)), ]
  list(gene = gene, edges = edges, first = first,
       tissue = read_table(gene, "_tissue_table.csv"),
       pertis = read_table(gene, "_per_tissue_effects.csv"),
       vals   = read_table(gene, "_top_edge_values.csv"),
       netw   = read_table(gene, "_network_edges.csv"),
       perm   = read_table(gene, "_permutation.csv"),
       S      = stats::setNames(as.list(as.character(smry$value)), smry$key))
}

# panel a: mutated lines per tissue
panel_tissues <- function(g, max_rows = 9) {
  tt <- g$tissue[g$tissue$n_mutated > 0, c("tissue", "n_mutated", "n_wild_type", "informative")]
  if (!nrow(tt)) return(empty_panel("No mutated cell lines"))
  tt <- tt[order(-tt$n_mutated, -tt$n_wild_type), ]
  if (nrow(tt) > max_rows) {
    rest <- tt[max_rows:nrow(tt), ]
    tt <- rbind(tt[seq_len(max_rows - 1), ],
                data.frame(tissue = sprintf("%d other tissues", nrow(rest)), n_mutated = sum(rest$n_mutated),
                           n_wild_type = sum(rest$n_wild_type), informative = as.integer(any(rest$informative == 1))))
  }
  tt$lab  <- paste0(tt$n_mutated, " of ", tt$n_mutated + tt$n_wild_type)
  tt$name <- vapply(tt$tissue, wrap_text, "", width = 16)
  tt$name <- factor(tt$name, levels = rev(tt$name))
  tt$fill <- ifelse(tt$informative == 1, COL$bar, COL$bar_pale)
  p <- ggplot(tt, aes(x = n_mutated, y = name)) +
    geom_col(aes(fill = fill), width = 0.62) +
    geom_text(aes(label = lab), hjust = -0.12, size = 5.5 / PT, family = FONT, colour = "black") +
    scale_fill_identity() +
    scale_x_continuous(expand = expansion(mult = c(0, 0.8))) +
    scale_y_discrete(expand = expansion(add = max(0.6, (6 - nrow(tt)) / 2))) +      # few tissues: the bars stay slim
    coord_cartesian(clip = "off") +
    labs(x = "Mutated cell lines", y = NULL) +
    theme_fig() +
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
          axis.text.y = element_text(size = 5.5, lineheight = 0.85))
  if (any(tt$informative != 1))
    p <- p + labs(caption = "Pale: no wild-type line in that tissue\n(it does not enter the estimate)") +
    theme(plot.caption = element_text(size = 5.5, hjust = 0, colour = "grey30", lineheight = 0.9),
          plot.caption.position = "plot")
  p
}

# panel b: volcano of the direct partners
panel_volcano <- function(g) {
  d <- g$first
  d$cls <- factor(ifelse(d$alt, d$dir, "ns"), levels = c("down", "up", "ns"))
  lab <- top_per_side(d[d$alt, ], N_LABELS)
  lab$label <- lab$Gene_B
  ref <- d[d$Is_Reference == 1, ]
  p <- ggplot(d, aes(x = Delta, y = y)) +
    geom_vline(xintercept = 0, linewidth = LW * 0.7, colour = "grey88") +
    geom_point(data = d[!d$alt, ], aes(fill = cls), shape = 21, colour = "transparent", stroke = 0, size = 0.8, alpha = 0.8) +
    geom_point(data = d[d$alt, ], aes(fill = cls), shape = 21, colour = "white", stroke = 0.18, size = 1.5)
  if (nrow(ref))
    p <- p + geom_point(data = ref, aes(colour = "ref"), shape = 21, fill = NA, stroke = 0.35, size = 1.9) +
    scale_colour_manual(values = c(ref = "black"), labels = paste0("Reference neighbour of ", g$gene))
  if (nrow(lab)) p <- p + repel_labels(lab, max(abs(d$Delta)))
  p + geom_hline(yintercept = -log10(FDR_CUT), linetype = "dashed", linewidth = LW * 0.8, colour = "grey35") +
    annotate("text", x = Inf, y = -log10(FDR_CUT), label = paste0("FDR = ", FDR_CUT), hjust = 1.05, vjust = -0.5,
             size = 5.5 / PT, colour = "grey25", family = FONT) +
    scale_fill_manual(values = dir_cols, breaks = c("down", "up", "ns"), drop = FALSE,
                      labels = c(sprintf("%s (n = %d)", LAB_DOWN, sum(d$cls == "down")),
                                 sprintf("%s (n = %d)", LAB_UP, sum(d$cls == "up")), "Not significant")) +
    guides(fill = guide_legend(order = 1, ncol = 1, override.aes = list(size = 1.7, alpha = 1, colour = "white", stroke = 0.2)),
           colour = guide_legend(order = 2, override.aes = list(size = 1.7, fill = "white"))) +
    scale_x_continuous(labels = minus, expand = expansion(mult = 0.05)) +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0.015, 0.1))) +
    labs(x = delta_title(paste0("association score, ", g$gene, "-mutated vs wild type")), y = Y_VOLCANO) +
    theme_fig() + theme(legend.box = "horizontal", legend.box.just = "top")
}

# panel c: cell-line scores of the strongest partners
panel_values <- function(g) {
  if (!nrow(g$vals)) return(empty_panel("No partner values exported"))
  v <- g$vals
  v$score <- v$value_adjusted
  info <- g$first[match(unique(v$Gene_B), g$first$Gene_B), ]
  info$Gene_B <- factor(info$Gene_B, levels = info$Gene_B)
  v$Gene_B <- factor(v$Gene_B, levels = levels(info$Gene_B))
  v$grp <- ifelse(v$status == "WT", "ns", info$dir[match(v$Gene_B, info$Gene_B)])
  one <- v$Gene_B == levels(v$Gene_B)[1]
  xl <- c(sprintf("WT\n(%s)", fmt(sum(one & v$status == "WT"))), sprintf("Mutated\n(%s)", fmt(sum(one & v$status != "WT"))))
  v$xlab <- factor(ifelse(v$status == "WT", xl[1], xl[2]), levels = xl)
  info$line1 <- sprintf('Delta == %s"%.2f"', ifelse(info$Delta < 0, "-", ""), abs(info$Delta))
  info$line2 <- paste0("FDR == ", vapply(info$q, sci, ""))
  rng <- vapply(split(v$score, v$Gene_B), range, c(0, 0))                    # every facet has its own y axis
  info$y1 <- rng[2, ] + 0.27 * (rng[2, ] - rng[1, ])
  info$y2 <- rng[2, ] + 0.12 * (rng[2, ] - rng[1, ])
  ggplot(v, aes(x = xlab, y = score)) +
    geom_violin(aes(fill = grp), colour = NA, alpha = 0.45, width = 0.9, scale = "width") +
    geom_point(data = v[v$status != "WT", ], aes(colour = grp), position = position_jitter(width = 0.09, height = 0, seed = 1),
               size = 0.35, alpha = 0.75, stroke = 0) +
    geom_boxplot(width = 0.17, outlier.shape = NA, linewidth = LW * 0.8, fill = "white", colour = "black") +
    geom_text(data = info, aes(x = 0.5, y = y1, label = line1), parse = TRUE, hjust = 0, vjust = 0.5,
              size = 5.5 / PT, family = FONT, inherit.aes = FALSE) +
    geom_text(data = info, aes(x = 0.5, y = y2, label = line2), parse = TRUE, hjust = 0, vjust = 0.5,
              size = 5.5 / PT, family = FONT, inherit.aes = FALSE) +
    facet_wrap(~ Gene_B, nrow = 2, scales = "free_y") +
    scale_fill_manual(values = dir_cols, guide = "none") +
    scale_colour_manual(values = dir_cols, guide = "none") +
    scale_y_continuous(labels = minus, expand = expansion(mult = c(0.04, 0.08))) +
    labs(x = NULL, y = "Association score (tissue-adjusted)") +
    theme_fig() + theme(panel.spacing = unit(2, "mm"))
}

# panel d: effect inside each tissue
panel_heatmap <- function(g) {
  first  <- g$first
  pertis <- g$pertis
  genes <- if (nrow(pertis)) first$Gene_B[first$Gene_B %in% pertis$Gene_B] else first$Gene_B
  genes <- utils::head(genes, N_HEATMAP)
  if (!length(genes)) return(empty_panel("No partners to show"))
  pooled <- first[match(genes, first$Gene_B), ]
  blocks <- c("Pooled", "Within tissue (n mutated)")
  long <- data.frame(Gene_B = genes, column = "All tissues", block = blocks[1], Delta = pooled$Delta, mark = FALSE)
  col_levels <- "All tissues"
  if (nrow(pertis)) {
    pt  <- pertis[pertis$Gene_B %in% genes, ]
    ord <- unique(pt[, c("tissue", "n_mutated")])
    ord <- utils::head(ord[order(-ord$n_mutated, ord$tissue), ], N_TISSUES)
    pt  <- pt[pt$tissue %in% ord$tissue, ]
    lab <- stats::setNames(vapply(sprintf("%s (%d)", ord$tissue, ord$n_mutated), wrap_text, "", width = 19), ord$tissue)
    long <- rbind(long, data.frame(Gene_B = pt$Gene_B, column = unname(lab[pt$tissue]), block = blocks[2],
                                   Delta = pt$Delta, mark = !is.na(pt$p) & pt$p < 0.05))
    col_levels <- c(col_levels, unname(lab))
  }
  long$column <- factor(long$column, levels = col_levels)
  long$block  <- factor(long$block, levels = blocks)
  long$Gene_B <- factor(long$Gene_B, levels = pooled$Gene_B[order(pooled$Delta)])
  lim <- max(0.1, ceiling(stats::quantile(abs(long$Delta), 0.98, na.rm = TRUE) * 10) / 10)
  upright <- length(col_levels) > 8
  ggplot(long, aes(x = column, y = Gene_B, fill = Delta)) +
    geom_tile(colour = "white", linewidth = 0.35) +
    geom_point(data = long[long$mark, ], shape = 16, size = 0.45, colour = "black") +
    facet_grid(. ~ block, scales = "free_x", space = "free_x") +
    scale_fill_gradient2(low = COL$down, mid = COL$mid, high = COL$up, midpoint = 0, limits = c(-lim, lim),
                         oob = scales::squish, labels = minus, name = DELTA) +
    scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) +
    labs(x = NULL, y = NULL) +
    theme_fig() +
    theme(axis.line = element_blank(), axis.ticks = element_blank(),
          axis.text.x = element_text(angle = if (upright) 90 else 45, hjust = 1, vjust = if (upright) 0.5 else 1,
                                     size = 5.5, lineheight = 0.85),
          axis.text.y = element_text(size = 5.5),
          legend.position = "right", legend.direction = "vertical", legend.justification = "center",
          legend.title = element_text(size = 7), legend.key.height = unit(5, "mm"), legend.key.width = unit(2, "mm"),
          strip.clip = "off", panel.spacing = unit(1.5, "mm"), plot.margin = margin(5.2, 1, 1, 2.5, unit = "mm"))
}

# panel e: the gene and its altered partners
panel_network <- function(g) {
  first <- g$first
  nodes <- first[first$alt, ]
  if (nrow(nodes) < 3) nodes <- utils::head(first, min(12, nrow(first)))      # nothing passes: the smallest FDR are shown
  nodes <- utils::head(nodes, N_NETWORK)
  nodes <- nodes[order(nodes$Delta), ]
  n <- nrow(nodes)
  theta <- pi / 2 + 2 * pi * (seq_len(n) - 0.5) / n          # most lost just left of the top, most gained just right
  nodes$x <- cos(theta); nodes$y <- sin(theta)
  nodes$cls   <- ifelse(nodes$alt, nodes$dir, "ns")
  nodes$angle <- ifelse(nodes$x < 0, theta * 180 / pi + 180, theta * 180 / pi)
  nodes$hjust <- ifelse(nodes$x < 0, 1, 0)
  nodes$lw <- LW * (0.6 + 2.0 * abs(nodes$Delta) / max(abs(nodes$Delta)))
  p <- ggplot()
  ch <- g$netw[g$netw$Gene_A %in% nodes$Gene_B & g$netw$Gene_B %in% nodes$Gene_B, ]
  if (nrow(ch)) {
    ch$cls <- ifelse(ch$FDR < FDR_CUT, ifelse(ch$Delta > 0, "up", "down"), "ns")
    arcs <- do.call(rbind, lapply(seq_len(nrow(ch)), function(i) {
      a <- nodes[match(ch$Gene_A[i], nodes$Gene_B), ]; b <- nodes[match(ch$Gene_B[i], nodes$Gene_B), ]
      cbind(bezier(a$x, a$y, b$x, b$y), id = i, cls = ch$cls[i])
    }))
    p <- p + geom_path(data = arcs, aes(x = x, y = y, group = id), colour = "grey70", linewidth = LW * 0.7)
    if (any(arcs$cls != "ns"))
      p <- p + geom_path(data = arcs[arcs$cls != "ns", ], aes(x = x, y = y, group = id, colour = cls), linewidth = LW * 1.1,
                         alpha = 0.9)
  }
  p <- p +
    geom_segment(data = nodes, aes(x = 0, y = 0, xend = 0.95 * x, yend = 0.95 * y, colour = cls, linewidth = lw), alpha = 0.9) +
    geom_point(data = nodes, aes(x = x, y = y, fill = cls), shape = 21, size = 2.1, stroke = 0.3, colour = "white")
  if (any(nodes$Is_Reference == 1))
    p <- p + geom_point(data = nodes[nodes$Is_Reference == 1, ], aes(x = x, y = y), shape = 21, size = 2.1, stroke = 0.4,
                        colour = "black", fill = NA)
  # key drawn by hand, so that it keeps to two rows whatever the labels
  key <- function(p, col, row, text, draw) {
    x0 <- c(-1.56, 0.02)[col]; y0 <- c(-1.78, -1.95)[row]
    draw(p, x0, y0) + annotate("text", x = x0 + 0.24, y = y0, label = text, hjust = 0, size = 6 / PT, family = FONT)
  }
  line_key <- function(colour, lw) function(p, x0, y0)
    p + annotate("segment", x = x0, xend = x0 + 0.17, y = y0, yend = y0, colour = colour, linewidth = lw)
  p <- key(p, 1, 1, LAB_DOWN, line_key(COL$down, LW * 2))
  p <- key(p, 2, 1, LAB_UP, line_key(COL$up, LW * 2))
  if (any(nodes$Is_Reference == 1))
    p <- key(p, 1, 2, "Reference neighbour", function(p, x0, y0)
      p + annotate("point", x = x0 + 0.085, y = y0, shape = 21, size = 1.9, stroke = 0.4, colour = "black", fill = "white"))
  if (nrow(ch)) p <- key(p, 2, 2, "Reference edge", line_key("grey70", LW * 0.9))
  p + geom_text(data = nodes, aes(x = 1.09 * x, y = 1.09 * y, label = Gene_B, angle = angle, hjust = hjust), size = 5.5 / PT,
                family = FONT, colour = "black") +
    annotate("label", x = 0, y = 0, label = g$gene, fill = "grey15", colour = "white", size = 6 / PT, family = FONT,
             fontface = "bold", label.padding = unit(0.9, "mm"), label.r = unit(0.6, "mm")) +
    scale_colour_manual(values = dir_cols, guide = "none") +
    scale_fill_manual(values = dir_cols, guide = "none") +
    scale_linewidth_identity() +
    coord_fixed(xlim = c(-1.62, 1.62), ylim = c(-2.02, 1.62), clip = "off") +
    theme_void(base_size = 7, base_family = FONT) +
    theme(plot.margin = margin(5.2, 1, 1, 1, unit = "mm"))
}

# panel f: altered reference edges by degree
panel_degrees <- function(g) {
  edges <- g$edges
  ref <- edges[edges$Is_Reference == 1, ]
  lab <- c("1", "2", "3", "4+")
  dg <- do.call(rbind, lapply(1:4, function(k) {
    m <- ref$Degree == k; n <- sum(m); hit <- sum(ref$alt[m]); ci <- wilson(hit, n)
    null <- if (nrow(g$perm) && n > 0)
      100 * stats::quantile(g$perm[[paste0("n_sig_ref_degree", k)]], 0.975, names = FALSE) / n else NA_real_
    data.frame(xn = k, label = lab[k], n = n, pct = 100 * hit / max(n, 1), lo = 100 * ci[1], hi = 100 * ci[2], null = null)
  }))
  if (any(dg$n > 0 & dg$n < 20))
    message("Degree ", paste(dg$label[dg$n > 0 & dg$n < 20], collapse = ", "), " left out (fewer than 20 reference edges).")
  dg <- dg[dg$n >= 20, ]                      # a share of a handful of edges says nothing
  if (!nrow(dg)) return(empty_panel("Too few reference edges"))
  dg$label <- factor(dg$label, levels = lab)
  p <- ggplot(dg, aes(x = xn, y = pct))
  if (any(is.finite(dg$null)))
    p <- p + geom_segment(data = dg[is.finite(dg$null), ], aes(x = xn - 0.3, xend = xn + 0.3, y = null, yend = null, colour = "null"),
                          linewidth = LW * 1.6) +
    scale_colour_manual(values = c(null = "grey55"), labels = "Status shuffled (97.5th percentile)")
  p + geom_linerange(aes(ymin = lo, ymax = hi), linewidth = LW, colour = "black") +
    geom_point(aes(fill = label), shape = 21, size = 2, stroke = 0.3, colour = "black", show.legend = FALSE) +
    scale_fill_manual(values = COL$degree) +
    scale_x_continuous(breaks = dg$xn, labels = paste0(dg$label, "\n(", fmt_short(dg$n), ")"),
                       limits = c(0.5, max(dg$xn) + 0.5)) +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0.02, 0.08))) +
    labs(x = paste0("Degree from ", g$gene, " (edges tested)"), y = "Altered reference edges (%)") +
    theme_fig()
}

# panel g: with and without the tissue term
panel_covariate <- function(g) {
  d <- g$edges
  a <- d$FDR < FDR_CUT; u <- d$FDR_unadjusted < FDR_CUT
  d$cls <- factor(ifelse(a & u, "both", ifelse(u, "only_without", ifelse(a, "only_with", "ns"))),
                  levels = c("both", "only_without", "only_with", "ns"))
  n <- table(d$cls)
  lim <- max(abs(c(d$Delta, d$Delta_unadjusted)))
  cols <- c(both = COL$both, only_without = COL$only_without, only_with = COL$only_with, ns = COL$ns)
  p <- ggplot(d, aes(x = Delta_unadjusted, y = Delta, fill = cls)) +
    geom_abline(slope = 1, intercept = 0, linewidth = LW * 0.8, colour = "grey55", linetype = "dashed") +
    dense_points(data = d[d$cls == "ns", ], shape = 21, colour = "transparent", stroke = 0, size = 0.3, alpha = 0.6)
  for (k in c("both", "only_with", "only_without"))
    if (n[[k]] > 0) p <- p + dense_points(data = d[d$cls == k, ], shape = 21, colour = "transparent", stroke = 0, size = 0.55, alpha = 0.9)
  p + scale_fill_manual(values = cols, breaks = c("both", "only_without", "only_with"), drop = FALSE,
                        labels = c(sprintf("Altered in both (%s)", fmt(n[["both"]])),
                                   sprintf("Only without covariate (%s)", fmt(n[["only_without"]])),
                                   sprintf("Only with tissue covariate (%s)", fmt(n[["only_with"]])))) +
    guides(fill = guide_legend(ncol = 1, override.aes = list(size = 1.7, alpha = 1))) +
    scale_x_continuous(labels = minus, limits = c(-lim, lim)) + scale_y_continuous(labels = minus, limits = c(-lim, lim)) +
    labs(x = delta_title("(no covariate)"), y = delta_title("(tissue as covariate)")) +
    theme_fig() + theme(aspect.ratio = 1)
}

# panel h: the tissue with the largest weight left out
panel_one_out <- function(g) {
  d <- g$edges[is.finite(g$edges$Delta_without_top_tissue), ]
  if (!nrow(d)) return(empty_panel("Only one tissue holds\nboth groups"))
  d$cls <- factor(ifelse(d$alt, d$dir, "ns"), levels = c("down", "up", "ns"))
  n <- table(d$cls)
  a <- d[d$alt, ]
  lim <- max(abs(c(d$Delta, d$Delta_without_top_tissue)))
  note <- wrap_text(paste0(g$S[["heaviest_tissue"]], " left out"), 20)
  if (nrow(a) >= 3)
    note <- sprintf("%s\nsame sign: %s%%\nr = %.2f", note, round(100 * mean(sign(a$Delta) == sign(a$Delta_without_top_tissue)), 1),
                    stats::cor(a$Delta, a$Delta_without_top_tissue))
  p <- ggplot(d, aes(x = Delta, y = Delta_without_top_tissue, fill = cls)) +
    geom_abline(slope = 1, intercept = 0, linewidth = LW * 0.8, colour = "grey55", linetype = "dashed") +
    dense_points(data = d[d$cls == "ns", ], shape = 21, colour = "transparent", stroke = 0, size = 0.3, alpha = 0.6)
  for (k in c("down", "up"))
    if (n[[k]] > 0) p <- p + dense_points(data = d[d$cls == k, ], shape = 21, colour = "transparent", stroke = 0, size = 0.55, alpha = 0.9)
  p + annotate("text", x = -lim, y = lim, label = note, hjust = 0, vjust = 1, size = 5.5 / PT, family = FONT, lineheight = 0.95) +
    scale_fill_manual(values = dir_cols, breaks = c("down", "up", "ns"), drop = FALSE,
                      labels = c(sprintf("%s (%s)", LAB_DOWN, fmt(n[["down"]])),
                                 sprintf("%s (%s)", LAB_UP, fmt(n[["up"]])), "Not significant")) +
    guides(fill = guide_legend(ncol = 1, override.aes = list(size = 1.7, alpha = 1))) +
    scale_x_continuous(labels = minus, limits = c(-lim, lim)) + scale_y_continuous(labels = minus, limits = c(-lim, lim)) +
    labs(x = delta_title("all tissues"), y = delta_title("one tissue left out")) +
    theme_fig() + theme(aspect.ratio = 1)
}

# panel i: status shuffled within tissue
panel_permutation <- function(g) {
  if (!nrow(g$perm)) return(empty_panel("No permutations\nwere run"))
  null <- g$perm$n_sig
  obs <- sum(g$edges$alt)
  tr  <- function(v) log10(v + 1)
  top <- max(c(null, obs, 10))
  brk <- c(0, 10^(0:8)); brk <- brk[brk <= top]
  x_max <- tr(top) * 1.04 + 0.08
  left <- tr(obs) > 0.5 * x_max
  ggplot(data.frame(x = tr(null)), aes(x = x)) +
    geom_histogram(binwidth = 0.1, boundary = -0.05, fill = COL$bar, colour = "white", linewidth = 0.2) +
    geom_vline(xintercept = tr(obs), colour = COL$degree[["1"]], linewidth = LW * 1.6) +
    annotate("text", x = tr(obs), y = Inf, label = paste0("Observed: ", fmt(obs)), hjust = if (left) 1.08 else -0.08, vjust = 1.4,
             size = 5.5 / PT, family = FONT, colour = "black") +
    scale_x_continuous(breaks = tr(brk), labels = parse(text = ifelse(brk < 1000, as.character(brk), paste0("10^", round(log10(brk)))))) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.16))) +
    coord_cartesian(xlim = c(-0.08, x_max)) +
    labs(x = paste0("Altered edges (FDR < ", FDR_CUT, ")"), y = paste0("Permutations (of ", fmt(length(null)), ")")) +
    theme_fig()
}

# one panel by its name in the analysis
make_panel <- function(g, key) {
  switch(key, a = panel_tissues(g), b = panel_volcano(g), c = panel_values(g), d = panel_heatmap(g),
         e = panel_network(g), f = panel_degrees(g), g = panel_covariate(g), h = panel_one_out(g),
         i = panel_permutation(g))
}

panel_title <- function(g, key) {
  switch(key,
         a = "Mutated lines by tissue",
         b = paste0(g$gene, " associations, tissue-adjusted (", fmt(nrow(g$first)), " partners)"),
         c = "Cell-line scores, strongest partners",
         d = paste0("Effect of ", g$gene, " mutation inside each tissue (dot: P < 0.05 within the tissue)"),
         e = if (sum(g$first$alt) >= 3) paste0(g$gene, " and its strongest altered partners")
         else paste0(g$gene, ": no altered partner (smallest FDR shown)"),
         f = "Altered edges by degree", g = "With and without tissue",
         h = "Heaviest tissue left out", i = "Status shuffled within tissue")
}

# output
open_pdf <- function(file, w, h) {
  if (capabilities("cairo")) grDevices::cairo_pdf(file, width = w, height = h, family = FONT)
  else grDevices::pdf(file, width = w, height = h, useDingbats = FALSE)
}
open_png <- function(file, w, h) {
  if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png(file, width = w, height = h, units = "in", res = 600)
  else grDevices::png(file, width = w, height = h, units = "in", res = 600, type = "cairo")
}
write_figure <- function(draw, name, width_mm, height_mm) {
  w <- width_mm / 25.4; h <- height_mm / 25.4
  open_pdf(file.path(out_dir, paste0(name, ".pdf")), w, h); draw(); grDevices::dev.off()
  open_png(file.path(out_dir, paste0(name, ".png")), w, h); draw(); grDevices::dev.off()
}

# a panel with its letter and its short title above it
draw_panel <- function(plot, title, letter = NULL) {
  print(plot, newpage = FALSE)
  y <- unit(1, "npc") - unit(3.4, "mm")
  if (!is.null(letter))
    grid.text(letter, x = unit(0.8, "mm"), y = y, just = c("left", "bottom"), gp = gpar(fontsize = 8, fontface = "bold", fontfamily = FONT))
  grid.text(title, x = unit(if (is.null(letter)) 1 else 4.8, "mm"), y = y, just = c("left", "bottom"),
            gp = gpar(fontsize = 7, fontfamily = FONT))
}

# one panel of the analysis, `layout` is its entry in FIG5, SUPP6 or SUPP7
export_panel <- function(g, key, name, layout) {
  plot  <- make_panel(g, key)
  title <- panel_title(g, key)
  write_figure(function() { grid.newpage(); draw_panel(plot, title, layout$letter) },
               name, layout$box[3], layout$box[4])
}

# several panels on one page
export_page <- function(g, layouts, name) {
  panels <- lapply(names(layouts), function(k) make_panel(g, k))
  width  <- max(vapply(layouts, function(l) l$box[1] + l$box[3], 0))
  height <- max(vapply(layouts, function(l) l$box[2] + l$box[4], 0))
  draw <- function() {
    grid.newpage()
    for (i in seq_along(layouts)) {
      k <- names(layouts)[i]
      b <- layouts[[k]]$box
      pushViewport(viewport(x = unit(b[1], "mm"), y = unit(height - b[2], "mm"), width = unit(b[3], "mm"),
                            height = unit(b[4], "mm"), just = c("left", "top")))
      draw_panel(panels[[i]], panel_title(g, k), layouts[[k]]$letter)
      popViewport()
    }
  }
  write_figure(draw, name, width, height)
}
