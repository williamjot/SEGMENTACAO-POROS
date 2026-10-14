suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(png)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Uso: Rscript figuras.R <imagens> <validacao> <aplicacao> <saida>")
}

imagens_dir <- args[1]
validacao_dir <- args[2]
aplicacao_dir <- args[3]
saida_dir <- args[4]
dir.create(saida_dir, recursive = TRUE, showWarnings = FALSE)

validacao <- read.csv(file.path(validacao_dir, "metricas_por_patch.csv"))
aplicacao <- read.csv(file.path(aplicacao_dir, "resumo_por_patch.csv"))
aplicacao$diferenca <- aplicacao$porosidade_rf_percent - aplicacao$porosidade_rockface_percent

tema <- theme_classic(base_size = 10) +
  theme(panel.border = element_rect(fill = NA, colour = "black"), legend.position = "top")

p1 <- ggplot(validacao, aes(metodo, iou, colour = metodo)) +
  geom_jitter(width = 0.08, size = 2, alpha = 0.75) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3.5, fill = "white", colour = "black") +
  labs(x = NULL, y = "IoU nos pixels anotados") +
  guides(colour = "none") + tema

p2 <- ggplot(aplicacao, aes(porosidade_rockface_percent, porosidade_rf_percent, shape = origem)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
  geom_point(size = 2.7, colour = "#0065A3") +
  labs(x = "Área porosa pelo rockface (%)", y = "Área porosa pelo RF (%)", shape = NULL) + tema

p3 <- ggplot(aplicacao, aes(pixels_incertos_percent, iou_rf_rockface, shape = origem)) +
  geom_point(size = 2.7, colour = "#C45A00") +
  labs(x = "Pixels incertos (%)", y = "IoU RF x rockface", shape = NULL) + tema

p4 <- ggplot(
  aplicacao,
  aes(diametro_eq_mediana_px_original, densidade_poros_por_mpx, shape = origem)
) +
  geom_point(size = 2.7, colour = "#207A4B") +
  labs(
    x = "Diâmetro equivalente mediano (px originais)",
    y = "Densidade de poros inteiros (por Mpx)",
    shape = NULL
  ) + tema

ggsave(
  file.path(saida_dir, "resultados_quantitativos.pdf"),
  (p1 | p2) / (p3 | p4),
  width = 18.3,
  height = 13.2,
  units = "cm"
)

treino <- aplicacao$patch[aplicacao$origem == "treino"][1]
novos <- aplicacao[aplicacao$origem == "novo", ]
selecionados <- unique(c(
  treino,
  novos$patch[which.min(novos$pixels_incertos_percent)],
  novos$patch[which.max(novos$porosidade_rf_percent)],
  novos$patch[which.min(novos$iou_rf_rockface)]
))
selecionados <- selecionados[!is.na(selecionados)][1:min(4, length(selecionados))]

paleta <- colorRampPalette(c("white", "#D7E7F5", "#6BAED6", "#2171B5", "#08306B"))(256)

ler_original <- function(nome) {
  x <- readPNG(file.path(imagens_dir, paste0("patch_", nome, "_c0.png")), native = TRUE)
  passo <- max(1, floor(nrow(x) / 1024))
  y <- x[seq(1, nrow(x), by = passo), seq(1, ncol(x), by = passo)]
  class(y) <- class(x)
  y
}

ler_mascara <- function(nome) {
  x <- readPNG(file.path(aplicacao_dir, "mascaras", paste0("patch_", nome, "_poros.png")))
  as.raster(ifelse(x >= 0.5, "#0065A3", "#F2F2F2"))
}

ler_probabilidade <- function(nome) {
  x <- readPNG(file.path(aplicacao_dir, "probabilidades", paste0("patch_", nome, "_probabilidade.png")))
  indice <- pmax(1, pmin(256, floor(x * 255) + 1))
  as.raster(matrix(paleta[indice], nrow = nrow(x), ncol = ncol(x)))
}

pdf(file.path(saida_dir, "exemplos_qualitativos.pdf"), width = 7.2, height = 8.0)
grid.newpage()
layout <- grid.layout(length(selecionados) + 1, 3, heights = unit(c(0.12, rep(1, length(selecionados))), "null"))
pushViewport(viewport(layout = layout))

for (j in 1:3) {
  grid.text(c("Imagem PPL", "Máscara detectada", "Probabilidade")[j],
            vp = viewport(layout.pos.row = 1, layout.pos.col = j), gp = gpar(fontface = "bold"))
}

for (i in seq_along(selecionados)) {
  nome <- selecionados[i]
  imagens <- list(ler_original(nome), ler_mascara(nome), ler_probabilidade(nome))
  for (j in 1:3) {
    pushViewport(viewport(layout.pos.row = i + 1, layout.pos.col = j))
    grid.raster(imagens[[j]], width = unit(1, "npc"), height = unit(1, "npc"), interpolate = TRUE)
    grid.rect(gp = gpar(fill = NA, col = "black"))
    if (j == 1) {
      grid.text(nome, x = 0.03, y = 0.96, just = c("left", "top"),
                gp = gpar(col = "white", fontface = "bold", fontsize = 8))
    }
    popViewport()
  }
}

popViewport()
dev.off()

