package hector

import (
	"sync"
	"strconv"
)

func AlgorithmRun(classifier Classifier, train_path string, test_path string, params map[string]string) (float64, []LabelPrediction, error) {
	classifier.Init(params)
	
	steps, _ := strconv.ParseInt(params["steps"], 10, 64)
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := NewDataSet()
	var err error
	var wait sync.WaitGroup
	wait.Add(2)
	go func(){
		err = train_dataset.Load(train_path, global, int(steps))
		wait.Done()
	}()
	
	if err != nil{
		return 0.5, nil, err
	}
	
	go func(){
		classifier.Train(train_dataset)
		wait.Done()
	}()
	
	wait.Wait()
	
	wait.Add(2)
	test_dataset := DataSet{}
	test_dataset.Samples = make(chan *Sample, 1000)
	go func(){
		err = test_dataset.Load(test_path, global, 1)
		wait.Done()
	}()
	if err != nil{
		return 0.5, nil, err
	}
	
	predictions := []LabelPrediction{}
	go func(){
		for sample := range test_dataset.Samples {
			prediction := classifier.Predict(sample)
			predictions = append(predictions, LabelPrediction{Label: sample.Label, Prediction: prediction})
		}
		wait.Done()
	}()
	
	wait.Wait()
	
	auc := AUC(predictions)
	return auc, predictions, nil
}