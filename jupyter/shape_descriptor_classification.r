suppressPackageStartupMessages(library(e1071))
suppressPackageStartupMessages(library(kernlab))

src <- '../ects/'
norm <- 'Normalized Size'
d <- 158
TT <- 8
founders <- read.csv(paste(src,'combined_d',d,'_T',TT,'.csv',sep=''))

founders_names_original <- levels(unique(founders$Founder))
founders_names <- founders_names_original
founders_names[5] <- 'CA Mariout'
founders_names[11] <- 'Good Delta'
founders_names[17] <- 'Maison Carree'
founders_names[24] <- 'Palmella Blue'
founders_names[28] <- 'WI Winter'
print(founders_names)

dim(founders)

trad_traits <- colnames(founders)[10:20]
print(trad_traits)
topo_traits <- colnames(founders)[21:ncol(founders)]
print(topo_traits[1:10])

dims <- 12
kernel <- 'laplacedot'
kpar <- list(sigma=1)
kpc <- kernlab::kpca(~.,data=founders[,topo_traits], kernel=kernel, features=dims, kpar=kpar)

info_type <- 'Combined'
mixed <- cbind(founders[,trad_traits], kpc@rotated)
scaled_data <- base::scale(mixed, center=TRUE, scale=TRUE)
dim(scaled_data)
scaled_data[1:5, ]

percent_train <- 0.8
train_ids <- c()
for(i in 1:length(founders_names_original)){
    seed_ids <- which(founders$Founder == founders_names_original[i])
    seed_train <- sample(seed_ids, size = floor(length(seed_ids)*percent_train), 
                         replace=FALSE)
    train_ids <- c(train_ids, seed_train)
}
test_ids <- setdiff(1:nrow(founders), train_ids)
train_labs <- founders$Founder[train_ids]
test_labs <- founders$Founder[test_ids]

model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                    kernel='polynomial', coef0=25, degree=2, gamma=0.1, cost=200, scale=FALSE)
pred <- stats::predict(model, scaled_data[test_ids,])
clasification <- matrix(as.numeric(table(pred, test_labs)),
                        length(founders_names), length(founders_names))
accuracy <- sum(diag(clasification))/sum(clasification)

print(paste('Classification accuracy:', 100*signif(accuracy,3), '%'))

model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                    kernel='polynomial', coef0=5, degree=3, gamma=0.01, cost=150, scale=FALSE)
pred <- stats::predict(model, scaled_data[test_ids,])
clasification <- matrix(as.numeric(table(pred, test_labs)),
                        length(founders_names), length(founders_names))
accuracy <- sum(diag(clasification))/sum(clasification)

print(paste('Classification accuracy:', 100*signif(accuracy,3), '%'))

model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                    kernel='radial', gamma=0.01, cost=100, scale=FALSE)
pred <- stats::predict(model, scaled_data[test_ids,])
clasification <- matrix(as.numeric(table(pred, test_labs)),
                        length(founders_names), length(founders_names))
accuracy <- sum(diag(clasification))/sum(clasification)

print(paste('Classification accuracy:', 100*signif(accuracy,3), '%'))

info_type <- 'Combined'
mixed <- cbind(founders[,trad_traits], kpc@rotated)
scaled_data <- base::scale(mixed, center=TRUE, scale=TRUE)

sample_runs <- 100
percent_train <- 0.8
results <- base::array(0, dim=c(length(founders_names), length(founders_names), sample_runs))

for(j in 1:sample_runs){
    train_ids <- c()
    for(i in 1:length(founders_names_original)){
        seed_ids <- which(founders$Founder == founders_names_original[i])
        seed_train <- sample(seed_ids, size = floor(length(seed_ids)*percent_train), 
                             replace=FALSE)
        train_ids <- c(train_ids, seed_train)
    }
    test_ids <- setdiff(1:nrow(founders), train_ids)
    train_labs <- founders$Founder[train_ids]
    test_labs <- founders$Founder[test_ids]

    model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                        kernel='polynomial', coef0=5, degree=3, gamma=0.01, cost=50, scale=FALSE)
    pred <- stats::predict(model, scaled_data[test_ids,])
    results[,,j] <- matrix(as.numeric(table(pred, test_labs)),
                            length(founders_names), length(founders_names))
}

filename <- paste('svm_results',tolower(gsub(' ', '_', norm)), 
                  tolower(info_type), d, TT, kernel, dims, 'founders.rds', sep='_')
filename <- paste(src, filename, sep='')
base::saveRDS(results, filename)

info_type <- 'Topological'
scaled_data <- base::scale(kpc@rotated, center=TRUE, scale=TRUE)
dim(scaled_data)

percent_train <- 0.8
train_ids <- c()
for(i in 1:length(founders_names_original)){
    seed_ids <- which(founders$Founder == founders_names_original[i])
    seed_train <- sample(seed_ids, size = floor(length(seed_ids)*percent_train), 
                         replace=FALSE)
    train_ids <- c(train_ids, seed_train)
}
test_ids <- setdiff(1:nrow(founders), train_ids)
train_labs <- founders$Founder[train_ids]
test_labs <- founders$Founder[test_ids]

model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                    kernel='polynomial', coef0=10, degree=2, gamma=5, cost=100, scale=FALSE)
pred <- stats::predict(model, scaled_data[test_ids,])
clasification <- matrix(as.numeric(table(pred, test_labs)),
                        length(founders_names), length(founders_names))
accuracy <- sum(diag(clasification))/sum(clasification)

print(paste('Classification accuracy:', 100*signif(accuracy,3), '%'))

model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                    kernel='radial', coef0=10, degree=2, gamma=5, cost=100, scale=FALSE)
pred <- stats::predict(model, scaled_data[test_ids,])
clasification <- matrix(as.numeric(table(pred, test_labs)),
                        length(founders_names), length(founders_names))
accuracy <- sum(diag(clasification))/sum(clasification)

print(paste('Classification accuracy:', 100*signif(accuracy,3), '%'))

info_type <- 'Topological'
scaled_data <- base::scale(kpc@rotated, center=TRUE, scale=TRUE)

sample_runs <- 100
percent_train <- 0.8
results <- base::array(0, dim=c(length(founders_names), length(founders_names), sample_runs))

for(j in 1:sample_runs){
    train_ids <- c()
    for(i in 1:length(founders_names_original)){
        seed_ids <- which(founders$Founder == founders_names_original[i])
        seed_train <- sample(seed_ids, size = floor(length(seed_ids)*percent_train), 
                             replace=FALSE)
        train_ids <- c(train_ids, seed_train)
    }
    test_ids <- setdiff(1:nrow(founders), train_ids)
    train_labs <- founders$Founder[train_ids]
    test_labs <- founders$Founder[test_ids]

    model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                        kernel='polynomial', coef0=10, degree=2, gamma=5, cost=100, scale=FALSE)
    pred <- stats::predict(model, scaled_data[test_ids,])
    results[,,j] <- matrix(as.numeric(table(pred, test_labs)),
                            length(founders_names), length(founders_names))
}

filename <- paste('svm_results',tolower(gsub(' ', '_', norm)), 
                  tolower(info_type), d, TT, kernel, dims, 'founders.rds', sep='_')
filename <- paste(src, filename, sep='')
base::saveRDS(results, filename)

info_type <- 'Traditional'
scaled_data <- base::scale(founders[,trad_traits], center=TRUE, scale=TRUE)
dim(scaled_data)

mixed <- cbind(pca$rotation[,1:3], kpc@rotated)
scaled_data <- base::scale(mixed, center=TRUE, scale=TRUE)

percent_train <- 0.8
train_ids <- c()
for(i in 1:length(founders_names_original)){
    seed_ids <- which(founders$Founder == founders_names_original[i])
    seed_train <- sample(seed_ids, size = floor(length(seed_ids)*percent_train), 
                         replace=FALSE)
    train_ids <- c(train_ids, seed_train)
}
test_ids <- setdiff(1:nrow(founders), train_ids)
train_labs <- founders$Founder[train_ids]
test_labs <- founders$Founder[test_ids]

model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                    kernel='polynomial', coef0=5, degree=3, gamma=0.01, cost=50, scale=FALSE)
pred <- stats::predict(model, scaled_data[test_ids,])
clasification <- matrix(as.numeric(table(pred, test_labs)),
                        length(founders_names), length(founders_names))
accuracy <- sum(diag(clasification))/sum(clasification)

print(paste('Classification accuracy:', 100*signif(accuracy,3), '%'))

model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                    kernel='radial', coef0=10, degree=2, gamma=0.1, cost=10, scale=FALSE)
pred <- stats::predict(model, scaled_data[test_ids,])
clasification <- matrix(as.numeric(table(pred, test_labs)),
                        length(founders_names), length(founders_names))
accuracy <- sum(diag(clasification))/sum(clasification)

print(paste('Classification accuracy:', 100*signif(accuracy,3), '%'))

model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                    kernel='linear', coef0=10, degree=2, gamma=0.1, cost=10, scale=FALSE)
pred <- stats::predict(model, scaled_data[test_ids,])
clasification <- matrix(as.numeric(table(pred, test_labs)),
                        length(founders_names), length(founders_names))
accuracy <- sum(diag(clasification))/sum(clasification)

print(paste('Classification accuracy:', 100*signif(accuracy,3), '%'))

sample_runs <- 100
percent_train <- 0.8
results <- base::array(0, dim=c(length(founders_names), length(founders_names), sample_runs))

for(j in 1:sample_runs){
    train_ids <- c()
    for(i in 1:length(founders_names_original)){
        seed_ids <- which(founders$Founder == founders_names_original[i])
        seed_train <- sample(seed_ids, size = floor(length(seed_ids)*percent_train), 
                             replace=FALSE)
        train_ids <- c(train_ids, seed_train)
    }
    test_ids <- setdiff(1:nrow(founders), train_ids)
    train_labs <- founders$Founder[train_ids]
    test_labs <- founders$Founder[test_ids]

    model <- e1071::svm(scaled_data[train_ids,], train_labs, type='C-classification',
                        kernel='polynomial', coef0=10, degree=2, gamma=0.1, cost=10, scale=FALSE)
    pred <- stats::predict(model, scaled_data[test_ids,])
    results[,,j] <- matrix(as.numeric(table(pred, test_labs)),
                            length(founders_names), length(founders_names))
}

filename <- paste('svm_results',tolower(gsub(' ', '_', norm)), 
                  tolower(info_type), d, TT, kernel, dims, 'founders.rds', sep='_')
filename <- paste(src, filename, sep='')
base::saveRDS(results, filename)
