package main

import(
	"hector"
	"strconv"
	"fmt"
	"runtime/pprof"
	"os"
	"log"
)

func SplitFile(dataset *hector.DataSet, total, part int) (*hector.DataSet, *hector.DataSet) {

	train := hector.NewDataSet()
	test := hector.NewDataSet()

	for i, sample := range dataset.Samples {
		if i % total == part {
			test.AddSample(sample)
		} else {
			train.AddSample(sample)
		}
	}
	return train, test
}

func main(){
	train_path, _, _, method, params := hector.PrepareParams()
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	profile, _ := params["profile"]
	dataset := hector.NewDataSet()
	dataset.Load(train_path, global)
	
	cv, _ := strconv.ParseInt(params["cv"], 10, 32)
	total := int(cv)

	if profile != "" {
		f, err := os.Create(profile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	
	average_accuracy := 0.0
	for part := 0; part < total; part++ {
		train, test := SplitFile(dataset, total, part)
		classifier := hector.GetMutliClassClassifier(method)
		classifier.Init(params)
		accuracy := hector.MultiClassRunOnDataSet(classifier, train, test, "", params)
		fmt.Println("accuracy : ", accuracy)
		average_accuracy += accuracy
		classifier = nil
	}
	fmt.Println(average_accuracy / float64(total))
}