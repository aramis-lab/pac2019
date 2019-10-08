library(lme4)
library(lmtest)

donnees <- read.table(paste0(inrepertoire, '../src/validation/Dat_iter_0.tsv'), header = T, sep = '\t')

algos <- c('agedif_ggnet','agedif_resnet','agedif_bbnet','agedif_obbnet','agedif_svm','agedif_lmmmean','agedif_lmmquant')

pvalues = NULL
for (algo in algos){
  algo_p = NULL
  print(algo)
  donnees_abs = data.frame(donnees)
  #donnees_abs[, algo] <- abs(donnees_abs[, algo])
  model <- lmer(as.formula(paste0(algo ,'~ age + (1|gender) + (1|site)')), data = donnees, REML=F, na.action = "na.omit")
  model_age <- lmer(as.formula(paste0(algo ,'~ (1|gender) + (1|site)')), data = donnees, REML=F, na.action = "na.omit")
  model_site <- lmer(as.formula(paste0(algo ,'~ age + (1|gender)')), data = donnees, REML=F, na.action = "na.omit")
  model_gender <- lmer(as.formula(paste0(algo ,'~ age + (1|site)')), data = donnees, REML=F, na.action = "na.omit")
  
  algo_p = c(lrtest(model, model_age)$`Pr(>Chisq)`[2], lrtest(model, model_site)$`Pr(>Chisq)`[2], lrtest(model, model_gender)$`Pr(>Chisq)`[2])
  pvalues = rbind(pvalues, algo_p)
}

rownames(pvalues) <- algos
colnames(pvalues) <- c('age', 'site', 'gender')
pvalues <- t(pvalues)[,c('agedif_lmmmean', 'agedif_lmmquant', 'agedif_svm', 'agedif_bbnet', 'agedif_obbnet', 'agedif_resnet', 'agedif_ggnet')]

pvalues
pvalues < (0.001/21)

