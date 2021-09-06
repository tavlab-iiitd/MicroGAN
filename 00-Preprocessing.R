xdata<-readLines("C:\\Users\\saadf\\Downloads\\GSE37250_series_matrix.txt")
index<-grep("series_matrix_table",xdata)
matrix_tab<-xdata[(index[1]+1):(index[2]-1)]
writeLines(matrix_tab, "C:\\Users\\saadf\\Downloads\\GSE37250_matrix.txt")
matrix_data<-read.table("C:\\Users\\saadf\\Downloads\\GSE37250_matrix.txt",
                        sep = "\t", stringsAsFactors = FALSE, header = TRUE)
probeIDS <- matrix_data$ID_REF
expr_data<- matrix_data[,-1]
rownames(expr_data)<-matrix_data[,1]
offset_value<-(-min(expr_data))+1
expr_data <- expr_data+offset_value
expr_log2<-log2(expr_data)
save(expr_log2, file = "C:\\Users\\saadf\\Downloads\\GSE37250_log2.RData")
probeData<-readLines("C:\\Users\\saadf\\Downloads\\GPL10558_HumanHT-12_V4_0_R2_15002873_B.txt")
probeIndex<-grep("\\[Probes\\]", probeData)
controlIndex<-grep("\\[Controls\\]", probeData)
probeTab<-probeData[(probeIndex[1]+1):(controlIndex[1]-1)]
writeLines(probeTab, "C:\\Users\\saadf\\Downloads\\GPL10558_HumanHT-12_V4_0_R2_15002873_B_Probes.txt")
rm(list = ls())
probe_data<-read.delim("C:\\Users\\saadf\\Downloads\\GPL10558_HumanHT-12_V4_0_R2_15002873_B_Probes.txt", 
           sep = "\t", stringsAsFactors = FALSE, header = TRUE)
load("C:\\Users\\saadf\\Downloads\\GSE37250_log2.RData")
index_pass<-probe_data$Species!="ILMN Controls"
probe_pass<-probe_data[index_pass,]
expr_pass<-expr_log2[probe_pass$Probe_Id,]
index_symbol<-nchar(probe_pass$Symbol)!=0
expr_symbol<-expr_pass[index_symbol,]
probe_symbol<-probe_pass[index_symbol,]

