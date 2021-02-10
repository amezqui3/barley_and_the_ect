suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(viridis))
suppressPackageStartupMessages(library(RColorBrewer))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(dplyr))

source('descriptor_analysis_utils.R')

src <- '../ects/'
d <- 158
TT <- 8
kernel <- 'laplacedot'
dims <- 12
norm <- 'Normalized Size'

founders <- read.csv(paste(src, 'ect_d74_T4.csv',sep=''))
founders_names_original <- levels(unique(founders$Founder))
founders_names <- founders_names_original
founders_names[5] <- 'CA Mariout'
founders_names[11] <- 'Good Delta'
founders_names[17] <- 'Maison Carree'
founders_names[24] <- 'Palmella Blue'
founders_names[28] <- 'WI Winter'
print(founders_names)

info_type <- 'Combined'
filename <- paste('svm_results',tolower(gsub(' ', '_', norm)), 
                  tolower(info_type), d, TT, kernel, dims, 'founders.rds', sep='_')
filename <- paste(src, filename, sep='')
results <- base::readRDS(filename)
foolist <- get_confusion(results)
mixed_confusion <- foolist[[1]]
mixed_accuracy <- foolist[[2]]

options(repr.plot.width=15, repr.plot.height=1.5)
p <- plot_accuracy(mixed_accuracy, d, TT, kernel, dims, info_type, norm)
p

options(repr.plot.width=7.5, repr.plot.height=6.75)
p <- plot_confusion(mixed_confusion, founders_names, dim(results)[3], 'single', 
                    d, TT, kernel, dims, info_type, nor)
p

info_type <- 'Topological'
filename <- paste('svm_results',tolower(gsub(' ', '_', norm)), 
                  tolower(info_type), d, TT, kernel, dims, 'founders.rds', sep='_')
filename <- paste(src, filename, sep='')
results <- base::readRDS(filename)
foolist <- get_confusion(results)
topo_confusion <- foolist[[1]]
topo_accuracy <- foolist[[2]]

options(repr.plot.width=15, repr.plot.height=1.5)
p <- plot_accuracy(topo_accuracy, d, TT, kernel, dims, info_type, norm)
p

options(repr.plot.width=7.5, repr.plot.height=6.75)
p <- plot_confusion(topo_confusion, founders_names, dim(results)[3], 'single', 
                    d, TT, kernel, dims, info_type, norm)
p

info_type <- 'Traditional'
filename <- 'svm_results_traditional_founders.rds'
filename <- paste(src, filename, sep='')
results <- base::readRDS(filename)
foolist <- get_confusion(results)
trad_confusion <- foolist[[1]]
trad_accuracy <- foolist[[2]]

options(repr.plot.width=15, repr.plot.height=1.5)
p <- plot_accuracy(trad_accuracy, d, TT, kernel, dims, info_type, norm)
p

options(repr.plot.width=7.5, repr.plot.height=6.75)
p <- plot_confusion(trad_confusion, founders_names, dim(results)[3], 'single', 
                    d, TT, kernel, dims, info_type, norm)
p

signif(quantile(trad_accuracy, c(0.25, 0.5, 0.75)),3)*100
signif(quantile(topo_accuracy, c(0.25, 0.5, 0.75)),3)*100
signif(quantile(mixed_accuracy, c(0.25, 0.5, 0.75)),3)*100

signif(mean(trad_accuracy)*100, 3)
signif(mean(topo_accuracy)*100, 3)
signif(mean(mixed_accuracy)*100, 3)

info_type <- 'Combined vs Traditional'
options(repr.plot.width=7.5, repr.plot.height=6.5)
conf_diff <- (mixed_confusion - trad_confusion)/dim(results)[3]
rownames(conf_diff) <- founders_names
colnames(conf_diff) <- founders_names

p <- plot_comparison(conf_diff, founders_names, d, TT, kernel, dims, info_type, norm)
p

acc <- round(diag(conf_diff)*100, 2)
print(sort(acc))
paste('Median diag:', median(acc))
paste('Mean diag:', round(mean(acc),2))

info_type <- 'Combined vs Topological'
options(repr.plot.width=7.5, repr.plot.height=6.5)
conf_diff <- (mixed_confusion - topo_confusion)/dim(results)[3]
rownames(conf_diff) <- founders_names
colnames(conf_diff) <- founders_names

p <- plot_comparison(conf_diff, founders_names, d, TT, kernel, dims, info_type, norm)
p

acc <- round(diag(conf_diff)*100, 2)
print(sort(acc))
paste('Median diag:', median(acc))
paste('Mean diag:', round(mean(acc),2))

info_type <- 'Traditional vs Topological'
options(repr.plot.width=7.5, repr.plot.height=6.5)
conf_diff <- (trad_confusion - topo_confusion)/dim(results)[3]
rownames(conf_diff) <- founders_names
colnames(conf_diff) <- founders_names

p <- plot_comparison(conf_diff, founders_names, d, TT, kernel, dims, info_type, norm)
p

acc <- round(diag(conf_diff)*100, 2)
print(sort(acc))
paste('Median diag:', median(acc))
paste('Mean diag:', round(mean(acc),2))

svm_scores <- matrix(0,ncol=3,nrow=length(founders_names))
base::rownames(svm_scores) <- founders_names
base::colnames(svm_scores) <- c('Traditional', 'Topological', 'Combined')
svm_scores[,1] <- diag(trad_confusion)
svm_scores[,2] <- diag(topo_confusion)
svm_scores[,3] <- diag(mixed_confusion)
df_original <- data.frame(svm_scores)
df_original$Line <- founders_names

options(repr.plot.width=18, repr.plot.height=4)
p <- compare_descriptor_accuracies(df_original, 'Combined', d, TT, kernel, dims, norm, TRUE)
#p <- compare_descriptor_accuracies(df, 'Topological', d, TT, kernel, dims, norm)
#p <- compare_descriptor_accuracies(df, 'Combined', d, TT, kernel, dims, norm)
p
