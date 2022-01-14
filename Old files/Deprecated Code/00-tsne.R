rm(list=ls())
tsne.files <- list.files('./data/tsne_embeddings/',full.names = TRUE)
datList <- list()
for(i in 1:length(tsne.files)){
  dat <- read.csv(tsne.files[i],stringsAsFactors = FALSE,row.names = 1)
  xout <- strsplit(tsne.files[i],split = '\\_')[[1]]
  algo <- xout[grep('embedded',xout)+1]
  samp <- gsub('\\.csv','',xout[grep('.csv',xout)])
  dat$algo <- toupper(algo)
  dat$group <- samp
  datList[[i]] <- dat
}

datDF <- plyr::ldply(datList)

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
      geom_point() +
      facet_grid(algo~group,scale='free') +
      theme_pubr()
    k <- k + 1
  }
}

library(gridExtra)
grid.arrange(
  pout[[1]],pout[[3]],pout[[5]],
  pout[[2]],pout[[4]],pout[[6]],
  ncol=3
)