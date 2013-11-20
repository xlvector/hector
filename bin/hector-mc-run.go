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
		accuracy, _ := hector.MultiClassRun(classifier, train, test, pred, params)
		fmt.Println("accuracy : ", accuracy)
	} else if action == "train" {
		hector.MultiClassTrain(classifier, train, params)

	} else if action == "test" {
		accuracy, _ := hector.MultiClassTest(classifier, test, pred, params)
		fmt.Println("accuracy", accuracy)
	}
}