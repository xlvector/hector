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
	
	average_auc := 0.0
	for part := 0; part < total; part++ {
		train, test := SplitFile(dataset, total, part)
		classifier := hector.GetClassifier(method)
		classifier.Init(params)
		auc, _ := hector.AlgorithmRunOnDataSet(classifier, train, test, "", params)
		fmt.Println("AUC:")
		fmt.Println(auc)
		average_auc += auc
		classifier = nil
	}
	fmt.Println(average_auc / float64(total))
}