package main

import (
	"fmt"
	"github.com/xlvector/hector"
	"log"
	"os"
	"runtime/pprof"
)

func main() {
	train, test, pred, method, params := hector.PrepareParams()

	action, _ := params["action"]

	classifier := hector.GetMutliClassClassifier(method)

	profile, _ := params["profile"]
	if profile != "" {
		fmt.Printf("Profile data => %s\n", profile)
		f, err := os.Create(profile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

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
