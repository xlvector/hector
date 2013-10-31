hector
======

Golang machine learning lib

# Supported Algorithms

Logistic Regression (SGD, FTRL)
Factorized Machine
CART, Random Forest, Random Decision Tree

# How to Run

cd src

go build hector-cv.go

./hector-cv --method [Method] --train [Data Path]

Here, Method include [lr, ftrl, dt, rf, fm, rdt]

Data Path is location of your dataset, we support LibSVM data format
