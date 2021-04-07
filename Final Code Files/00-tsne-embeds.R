rm(list=ls())
tsne.files <- list.files('./data/embeds_non/',full.names = TRUE)
datList <- list()
for(i in 1:length(tsne.files)){
  dat <- read.csv(tsne.files[i],stringsAsFactors = FALSE,row.names = 1)
  xout <- strsplit(tsne.files[i],split = '\\/')[[1]]
  xout <- strsplit(xout[length(xout)],split = '\\_')[[1]]
  algo <- xout[1]
  samp <- ifelse(
    gsub('\\.csv','',xout[grep('.csv',xout)])=='1',
    'Healthy','TB')
  dat$algo <- toupper(algo)
  dat$group <- samp
  datList[[i]] <- dat
}

datDF <- plyr::ldply(datList)
datDF$LABEL <- ifelse(
  grepl('ORG',datDF$label),
  'Original','Synthetic')
library(ggplot2)
library(ggpubr)
algos <- unique(datDF$algo)
k <- 1
pout <- list()
for(i in 1:length(algos)){
  idx.algo <- datDF$algo == algos[i]
  algo.dat <- datDF[idx.algo,]
  types <- unique(algo.dat$group)
  for(j in 1:length(types)){
    types.dat <- algo.dat[algo.dat$group == types[j],]
    pout[[k]] <- ggplot(types.dat,aes(dim1,dim2)) +
      geom_density_2d(aes(col=LABEL),alpha=0.5) +
      geom_point(aes(col=LABEL)) +
      facet_grid(algo~group,scale='free') +
      theme_pubr()
    k <- k + 1
  }
}

library(gridExtra)
dir.create('./figures',showWarnings = FALSE)
pdf('./figures/00-tsne-embeds.pdf',width = 7.5,height = 6)
grid.arrange(
  pout[[1]],pout[[3]],pout[[5]],pout[[7]],
  pout[[2]],pout[[4]],pout[[6]],pout[[8]],
  ncol=4
)
dev.off()