package main

import (
	"fmt"
	"runtime"

	"github.com/xlvector/hector"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/lr"
)

func main() {
	train, test, pred, _, params := hector.PrepareParams()

	action, _ := params["action"]
	runtime.GOMAXPROCS(runtime.NumCPU())
	if action == "train" {
		classifier := &lr.LogisticRegressionStream{}
		data := core.NewStreamingDataSet()
		go data.Load(train, 1)
		classifier.Train(data)
		classifier.SaveModel(params["model"])
	} else if action == "test" {
		classifier := &lr.LogisticRegression{}
		auc, _, _ := hector.AlgorithmTest(classifier, test, pred, params)
		fmt.Println("AUC:")
		fmt.Println(auc)
	}
}
