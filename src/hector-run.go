package main

import(
	"hector"
	"fmt"
)

func main(){
	train, test, pred, method, params := hector.PrepareParams()
	
	classifier := hector.GetClassifier(method)
	
	auc, _, _ := hector.AlgorithmRun(classifier, train, test, pred, params)
	fmt.Println("AUC:")
	fmt.Println(auc)
}