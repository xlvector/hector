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

In src folder, you will find two program with main function : hector-cv.go and hector-run.go

hector-cv.go will help you test one algorithm by cross validation in some dataset, you can run it by following steps:

	cd src
	go build hector-cv.go
	./hector-cv --method [Method] --train [Data Path]

Here, Method include

1. lr : Logistic Regression with SGD and L2 regularization.
2. ftrl : FTRL-Proximal Logistic Regreesion with L1 regularization. Please review this paper for more details "Ad Click Prediction: a View from the Trenches".
3. ep : Bayesian Logistic Regression with expectation propagation. Please review this paper for more details "Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising in Microsoft’s Bing Search Engine"

Data Path is location of your dataset, we support LibSVM data format

For example, you can run 10-fold cross validation by random forest on test data by:

./hector-cv --train ../data/titanic --method rf --cv 10
