package main

import (
	"fmt"
	"github.com/hector"
)

func main() {
	train, test, pred, method, params := hector.PrepareParams()

	action, _ := params["action"]

	classifier := hector.GetClassifier(method)

	if action == "" {
		auc, _, _ := hector.AlgorithmRun(classifier, train, test, pred, params)
		fmt.Println("AUC:")
		fmt.Println(auc)
	} else if action == "train" {
		hector.AlgorithmTrain(classifier, train, params)

	} else if action == "test" {
		auc, _, _ := hector.AlgorithmTest(classifier, test, pred, params)
		fmt.Println("AUC:")
		fmt.Println(auc)
	}
}
