ReadORMBin=function(prefix, AllN=F, size=4){
  sum_i=function(i){
    return(sum(1:i))
  }
  BinFileName=paste(prefix,".orm.bin",sep="")
  NFileName=paste(prefix,".orm.N.bin",sep="")
  IDFileName=paste(prefix,".orm.id",sep="")
  id = read.table(IDFileName)
  n=dim(id)[1]
  BinFile=file(BinFileName, "rb");
  grm=readBin(BinFile, n=n*(n+1)/2, what=numeric(0), size=size)
  NFile=file(NFileName, "rb");
  if(AllN==T){
    N=readBin(NFile, n=n*(n+1)/2, what=numeric(0), size=size)
  }
  else N=readBin(NFile, n=1, what=numeric(0), size=size)
  i=sapply(1:n, sum_i)
  return(list(diag=grm[i], off=grm[-i], id=id, N=N))
}

# Format matrix
asBRM<-function(BRMbin){
  mat<-matrix(0, nrow = length(BRMbin$diag), ncol = length(BRMbin$diag))
mat[upper.tri(mat)]<-BRMbin$off
mat<-mat+t(mat)
diag(mat)<-BRMbin$diag
colnames(mat)<-BRMbin$id[,2]
rownames(mat)<-BRMbin$id[,2]
  return(mat)
}


# Run QC and export list of participants to exclude
QC_BRM_outliers<-function(BRM){

library(ggplot2)
library(ggsignif)

mat<-asBRM(ReadORMBin(BRM))
rs<-rowMeans(abs(mat))

png(paste0("Hist_BRM_rowSums_", BRM, ".png"),  width = 30,height = 30, units = "cm", res = 300)
hist(rs, breaks=100)
abline(v=mean(rs)+4*sd(rs))
dev.off()

ne2<-names(rs[which(rs> (mean(rs)+4*sd(rs)) ) ])
write.table(ne2, paste0("IDs_BRM_rowSums_outlier", BRM, ".txt"), col.names = F, row.names = F, quote = F )

png(paste0("Hist_BRM_offDiag_", BRM, ".png"),  width = 30,height = 30, units = "cm", res = 300)
hist(mat[upper.tri(mat)], breaks=100)
dev.off()

png(paste0("Hist_BRM_offDiag_Woutliers_", BRM, ".png"),  width = 30,height = 30, units = "cm", res = 300)
mat2<-mat[-which(colnames(mat) %in% ne2),-which(rownames(mat) %in% ne2)]
hist(mat2[upper.tri(mat2)], breaks=100)
dev.off()

png(paste0("Hist_BRM_offDiag_Woutliers_all", BRM, ".png"),  width = 30,height = 30, units = "cm", res = 300)
ne<-unique(c(colnames(mat)[which(diag(mat)>2.5)], ne2))
mat3<-mat[-which(colnames(mat) %in% ne),-which(rownames(mat) %in% ne)]
hist(mat3[upper.tri(mat3)], breaks=100)
write.table(ne, paste0("IDs_BRM_rowSums_outlier_diagOutlier", BRM, ".txt"), col.names = F, row.names = F, quote = F )

dev.off()


}
