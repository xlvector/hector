hector
======

Golang machine learning lib. Currently, it can be used to solve binary classification problems.

# Supported Algorithms

1. Logistic Regression
2. Factorized Machine
3. CART, Random Forest, Random Decision Tree, Gradient Boosting Decision Tree
4. Neural Network

# Dataset Format

Hector support libsvm-like data format. Following is an sample dataset

	1 	1:0.7 3:0.1 9:0.4
	0	2:0.3 4:0.9 7:0.5
	0	2:0.7 5:0.3
	...

# How to Run

## Run as tools

hector-cv.go will help you test one algorithm by cross validation in some dataset, you can run it by following steps:

	go get github.com/xlvector/hector
	go install github.com/xlvector/hector/hectorcv
	hectorcv --method [Method] --train [Data Path] --cv 10

Here, Method include

1. lr : logistic regression with SGD and L2 regularization.
2. ftrl : FTRL-proximal logistic regreesion with L1 regularization. Please review this paper for more details "Ad Click Prediction: a View from the Trenches".
3. ep : bayesian logistic regression with expectation propagation. Please review this paper for more details "Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising in Microsoft’s Bing Search Engine"
4. fm : factorization machine
5. cart : classifiaction tree
6. cart-regression : regression tree
7. rf : random forest
8. rdt : random decision trees
9. gbdt : gradient boosting decisio tree
10. linear-svm : linear svm with L1 regularization
11. svm : svm optimizaed by SMO (current, its linear svm)
12. l1vm : vector machine with L1 regularization by RBF kernel
13. knn : k-nearest neighbor classification

hector-run.go will help you train one algorithm on train dataset and test it on test dataset, you can run it by following steps:

	cd src
	go build hector-run.go
	./hector-run --method [Method] --train [Data Path] --test [Data Path]

Above methods will direct train algorithm on train dataset and then test on test dataset. If you want to train algorithm and get the model file, you can run it by following steps:

	./hector-run --method [Method] --action train --train [Data Path] --model [Model Path]

Then, you can use model file to test any test dataset:

	./hector-run --method [Method] --action test --test [Data Path] --model [Model Path]

# Benchmark

## Binary Classification

Following are datasets used in benchmarks, You can find them from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/)

1. heart
2. fourclass

I will do 5-fold cross validation on the dataset, and use AUC as evaluation metric. Following are the results:

DataSet | Method | AUC
------- | ------ | ---
heart   | FTRL-LR   |0.9109
heart   | EP-LR | 0.8982
heart | CART | 0.8231
heart | RDT | 0.9155
heart | RF | 0.9019
heart | GBDT | 0.9061
fourclass | FTRL-LR | 0.8281
fourclass | EP-LR | 0.7986
fourclass | CART | 0.9832
fourclass | RDT | 0.9925
fourclass | RF | 0.9947
fourclass | GBDT | 0.9958

