
## LOADING THE DATA
white.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
white.raw <- read.csv(white.url, header = TRUE, sep = ";")
white <- white.raw
##===========================================
##EDA
str(white)
dim(white)
summary(white)
boxplot(white)

par(mfrow = c(3,4))
for (i in c(1:11)) {
  plot(white[, i], jitter(white[, "quality"]), xlab = names(white)[i],
       ylab = "quality", col = "blue", cex = 0.8, cex.lab = 1.3)
  abline(lm(white[, "quality"] ~ white[ ,i]), lty = 2, lwd = 2)
}
par(mfrow = c(1, 1))

par(mfrow = c(1,1))
cor.white <- cor(white)
corrplot(cor.white, method = 'number')

par(mfrow = c(4,3))
for (i in c(1:12)) {
  hist(white[, i], xlab = names(white)[i], col = "blue", 
       breaks = 20, main = paste("Histogram of", names(white)[i]))
}
par(mfrow = c(1, 1))

hist(white$quality, col="#458B74", 
     main="Wine Quality Histogram", 
     xlab="Quality", 
     ylab="Number of samples")

table(white$quality, dnn = "No. of Samples")
quality_fac <- ifelse(white$quality >= 6, "0", "1")
white <- data.frame(white, quality_fac) 
table(white$quality_fac)
white <- white[,-12]
##===========================================
##Splitting the data
subset <- sample(nrow(white), nrow(white) * 0.8)
white.train = white[subset, ]
white.test = white[-subset, ]

dim(white.train)
dim(white.test)
summary(white.train)
##===========================================
## LOgistics Regression
white.glm0 <- glm(quality_fac ~ ., family = binomial, white.train)
summary(white.glm0)

white.glm.step <- step(white.glm0)
summary(white.glm.step)
white.prob.glm.step.insample <- predict(white.glm.step, type = "response")
white.predicted.glm.step.insample <- white.prob.glm.step.insample > 0.5
white.predicted.glm.step.insample <- as.numeric(white.predicted.glm.step.insample)

roc.plot(white.train$quality_fac == "1", white.prob.glm.step.insample)
roc.plot(white.train$quality_fac == "1", white.prob.glm.step.insample)$roc.vol

## Validating the Model
white.prob.glm1.outsample <- predict(white.glm.step, white.test, type = "response")
white.predicted.glm1.outsample <- white.prob.glm1.outsample > 0.5
white.predicted.glm1.outsample <- as.numeric(white.predicted.glm1.outsample)
table(credit.test$response, predicted.glm1.outsample, dnn = c("Truth", "Predicted"))

mean(ifelse(white.test$quality_fac != white.predicted.glm1.outsample, 1, 0))

roc.plot(white.test$quality_fac == "1", white.prob.glm1.outsample)$roc.vol
##===========================================
## CLASSIFICATIOON TREE MODEL

white.rpart.model <- rpart(quality_fac~., data = white.train, 
                           method="class",
                           parms = list(loss = matrix(c(0, 10, 1, 0), nrow = 2))) 
white.rpart.model
plot(white.rpart.model)
text(white.rpart.model, pretty = TRUE)

# MISCLASSIFICATION TABLE / CONFUSION MATRIX
white.test.prob.rpart = predict(white.rpart.model, white.test, type = "class")
table(white.test$quality_fac, white.test.prob.rpart, dnn = c("Truth", "Predicted"))

# MISCLASSIFICATION RATE
cost <- function(r, pi) { 
  weight1 = 10     
  weight0 = 1     
  c1 = (r == 1) & (pi == 0)  #logical vector - true if actual 1 but predict 0     
  c0 = (r == 0) & (pi == 1)  #logical vecotr - true if actual 0 but predict 1     
  return(mean(weight1 * c1 + weight0 * c0)) 
}
cost(white.test$quality_fac, white.test.prob.rpart)

# ROC and AUC
white.test.prob.rpart = predict(white.rpart.model, white.test, type = "prob")
pred.test = prediction(white.test.prob.rpart[, 2], white.test$quality_fac) 
perf = performance(pred.test, "tpr", "fpr")
plot(perf, colorize = TRUE)
slot(performance(pred.test, "auc"), "y.values")[[1]]
# class(white.test.prob.rpart)
##===========================================
## KNN

## Split up the x and y parts of the training data
y<-as.factor(white.train[ , 12])
x<-white.train[, 1:11]
x <- sapply(white.train[, 1:11], as.numeric)

#NORMALIZING THE training DATA
normalize<-function(numbers) {
  (numbers-mean(numbers))/sd(numbers)
}
## Apply this to our x data
x.normalized<-apply(x, 2, normalize)
## Check that it worked
apply(x.normalized, 2, mean)
apply(x.normalized, 2, sd)

## Loop through values of k
for(k in 1:19) {
  predicted<-knn.cv(x.normalized, y, k)
  print(paste("With", k, "neighbors the accuracy is", 
              sum(y== predicted)/nrow(x.normalized)))
}

#validation
y.test<-as.factor(white.test[ , 12])
x.test<-white.test[, 1:11]
x.test <- sapply(white.test[, 1:11], as.numeric)

## NNORMALIZING THE TESTING DATA
x.test.normalized <- apply(x.test, 2, normalize)
apply(x.test.normalized, 2, mean)
apply(x.test.normalized, 2, sd)

predicted<-knn(x.normalized, x.test.normalized, y, 19)
sum(y.test== predicted)/nrow(x.test.normalized)

## Confusion matrix
confusionMatrix(white.test$quality_fac, predicted, dnn = c("Truth", "Predicted"))

## Misclassification error
mean(ifelse(white.test$quality_fac != predicted, 1, 0))

## ROC and AUC
white.test.prob.knn = knn(train = x.normalized, test = x.test.normalized, cl = y, k = 19, prob = TRUE)
pred.test.knn = prediction(attr(white.test.prob.knn, "prob"), white.test$quality_fac) 
perf.knn = performance(pred.test.knn, "tpr", "fpr")
plot(perf.knn, colorize = TRUE)
slot(performance(pred.test.knn, "auc"), "y.values")[[1]]
# attr(white.test.prob.knn, "prob")
##===========================================

# CLUSTERING
# 
# white.cluster.fit <- kmeans(white.train[,1:11], 2)
# table(white.cluster.fit$cluster)
# plotcluster(white.train[, 1:11],white.cluster.fit$cluster)
# for (k in 1:2){
#   print(white.train$quality_fac[white.cluster.fit$cluster == k])
# }
# table(white.train$quality_fac[white.cluster.fit$cluster == 2])
# 
# white.dist <- dist(white.train[, 1:11])
# white.hclust = hclust(white.dist, method ="ward.D2")
# plot(white.hclust)
# 
# white.treecut = cutree(white.hclust, k = 2)
# table(white.treecut)
# for (k in 1:2){
#   print(white.train$quality_fac[white.treecut == k])
# }
# table(white.train$quality_fac[white.treecut == 2])



