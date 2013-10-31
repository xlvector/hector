hector
======

Golang machine learning lib

# Supported Algorithms

1. Logistic Regression (SGD, FTRL)
2. Factorized Machine
3. CART, Random Forest, Random Decision Tree

# How to Run

cd src

go build hector-cv.go

./hector-cv --method [Method] --train [Data Path]

Here, Method include [lr, ftrl, dt, rf, fm, rdt]

Data Path is location of your dataset, we support LibSVM data format

For example, you can run random forest on test data by:

./hector-cv --train ../data/titanic --method rf
