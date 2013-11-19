package main

import(
	"hector"
	"fmt"
)

func main(){
	train, test, pred, method, params := hector.PrepareParams()
	
	action, _ := params["action"]

	classifier := hector.GetMutliClassClassifier(method)
	
	if action == "" {
		auc, _, _ := hector.MultiClassRun(classifier, train, test, pred, params)
		fmt.Println("AUC:")
		fmt.Println(auc)
	} else if action == "train" {
		hector.MultiClassTrain(classifier, train, params)

	} else if action == "test" {
		auc, _, _ := hector.MultiClassTest(classifier, test, pred, params)
		fmt.Println("AUC:")
		fmt.Println(auc)
	}
}