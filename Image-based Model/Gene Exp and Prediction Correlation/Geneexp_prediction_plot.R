library(readxl)
library(dplyr)
library(ggplot2)
library(forcats)

df <- read_excel("Downloads/geneexp_prediction_corrcoeff.xlsx")

cell_lines <- c("U2OS", "NIHOVCAR3", "HEPG2")

moa_colors <- c(
  "AKT" = "skyblue",
  "AURORA KINASE" = "hotpink",
  "CDK" = "darkgreen",
  "EGFR" = "orange",
  "HDAC" = "purple",
  "HMGCR" = "brown",
  "JAK" = "lightgreen",
  "MTOR" = "red",
  "PARP" = "dimgray",
  "TOPOISOMERASE" = "navy",
  "TUBULIN" = "darkgoldenrod4",
  "TYROSINE KINASE" = "pink",
  "VEGFR" = "yellow"
)

ggplot(
  df %>% filter(CellLine %in% cell_lines),
  aes(
    x = DepMap_Expression,
    y = `Prediction Performance`,
    color = MOA
  )
) +
  # 🔹 Scatter points: filled circles with black edge
  geom_point(
    size = 2.8,
    alpha = 0.85,
    shape = 16,          # solid circle
    stroke = 0.3
  ) +
  
  # 🔹 Regression with shaded confidence band
  geom_smooth(
    method = "lm",
    se = TRUE,           # ⬅ enables grey shading
    color = "black",
    fill = "grey40",
    alpha = 0.35,
    linewidth = 0.9
  ) +
  
  facet_wrap(~ CellLine, nrow = 1) +
  scale_color_manual(values = moa_colors) +
  
  labs(
    title = "Gene Expression vs Prediction Performance",
    x = "Gene Expression (DepMap)",
    y = "Prediction Performance",
    color = "Mechanism of Action"
  ) +
  
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    
    # 🔹 Axis titles
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    
    # 🔹 Axis tick labels
    axis.text.x  = element_text(size = 12),
    axis.text.y  = element_text(size = 12),
    

    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey85"),
    legend.position = "right"
  ) +
  
  xlim(0, 9)

ggsave(
  "gene_expression_vs_prediction_scatter_plot.png",
  width = 12,
  height = 4,
  dpi = 400,
  bg = "white"
)


waterfall_df <- df %>%
  filter(
    CellLine %in% cell_lines,
    !is.na(Correlation_Coefficient)
  ) %>%
  group_by(Gene, MOA) %>%
  summarise(
    corr = mean(Correlation_Coefficient, na.rm = TRUE),
    .groups = "drop"
  )


ggplot(
  waterfall_df %>%
    arrange(desc(corr)) %>%
    mutate(Gene = fct_reorder(Gene, corr)),
  aes(x = Gene, y = corr, fill = MOA)
) +
  geom_col(width = 0.75) +
  scale_fill_manual(values = moa_colors) +
  
  labs(
    title = "Gene-level Correlation",
    subtitle = "Genes ranked by expression-prediction correlation",
    x = "Gene",
    y = "Correlation Coefficient",
    fill = "Mechanism of Action"
  ) +
  
  theme_bw(base_size = 13) +
  theme(
    # 🔹 Rotate gene names
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1,
      size = 10.5
    ),
    
    # 🔹 Axis titles
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    
    # 🔹 Axis tick labels
    axis.text.y  = element_text(size = 12),
    
    
    # 🔹 Reduce clutter
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    
    # 🔹 Smaller legend
    legend.title = element_text(size = 10),
    legend.text  = element_text(size = 9),
    legend.key.size = unit(0.35, "cm"),
    legend.position = "right"
  ) +
  
  geom_hline(yintercept = 0, linetype = "dashed", color = "black")

ggsave(
  "gene_level_correlation.png",
  width = 12,
  height = 4,
  dpi = 400,
  bg = "white"
)
