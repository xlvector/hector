package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/xlvector/hector"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/lr"
)

func main() {
	train, test, pred, _, params := hector.PrepareParams()
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	action, _ := params["action"]
	runtime.GOMAXPROCS(runtime.NumCPU())
	if action == "train" {
		classifier := &lr.LogisticRegressionStream{}
		classifier.Init(params)
		data := core.NewStreamingDataSet()
		go data.Load(train, 1)
		classifier.Train(data)
		classifier.SaveModel(params["model"])
	} else if action == "test" {
		classifier := &lr.LogisticRegression{}
		classifier.Init(params)
		auc, _, _ := hector.AlgorithmTest(classifier, test, pred, params)
		fmt.Println("AUC:")
		fmt.Println(auc)
	}
}
