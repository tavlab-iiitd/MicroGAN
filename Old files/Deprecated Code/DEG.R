library("optparse")

option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL, 
              help="dataset1 file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default=NULL, 
              help="dataset2 file name", metavar="character"),
    make_option(c("-s", "--start"), type="integer", default=NULL, 
              help="starting point", metavar="integer")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

get_stats = function(gene, group){
  df_gene_group = cbind(gene, group)
  t = t.test(gene~group, df_gene_group)
  t_val = data.frame(
    tscore = t$statistic, pval =  t$p.value,
    log2FC = t$estimate[2] - t$estimate[1]
  )
  return(t_val)
}

get_DE = function(data, group){
  output = list()
  for (i in 1:nrow(data)) {
    output[[i]] = get_stats(data[i,], group)
  }

  names(output) = rownames(data)
  output.df = plyr::ldply(output)
  output.df$adjpval = p.adjust(output.df$pval)
  return(output.df)
}

gp1 = read.csv(opt$file)
gp2 = read.csv(opt$out)
# print("files read")
# print(c(dim(gp1)[1], dim(gp1)[2]))
# print(c(dim(gp2)[1], dim(gp2)[2]))
gp1=gp1[,opt$start:dim(gp1)[2]]
gp2=gp2[,opt$start:dim(gp2)[2]]
# print(c(dim(gp1), dim(gp2)))
# print(c(dim(gp1)[1], dim(gp1)[2]))
# print(c(dim(gp2)[1], dim(gp2)[2]))
df_ = cbind(t(gp1), t(gp2))
grp_ = factor(rep(c("OrgGrp1", "OrgGrp2"), c(dim(gp1)[1], dim(gp2)[1])))

output.df = get_DE(df_, group = grp_)

print(output.df$.id[which(abs(output.df$log2FC) > 0.5 & output.df$adjpval< 0.05)])