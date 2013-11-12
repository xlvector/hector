package hector

import (
	"strconv"
	"os"
)

func AlgorithmRun(classifier Classifier, train_path string, test_path string, pred_path string, params map[string]string) (float64, []*LabelPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := NewDataSet()

	err := train_dataset.Load(train_path, global)
	
	if err != nil{
		return 0.5, nil, err
	}
	
	test_dataset := NewDataSet()
	err = test_dataset.Load(test_path, global)
	if err != nil{
		return 0.5, nil, err
	}
	
	auc, predictions := AlgorithmRunOnDataSet(classifier, train_dataset, test_dataset, pred_path, params)

	return auc, predictions, nil
}

func AlgorithmRunOnDataSet(classifier Classifier, train_dataset, test_dataset *DataSet, pred_path string, params map[string]string) (float64, []*LabelPrediction) {
	classifier.Init(params)
	
	classifier.Train(train_dataset)
	
	predictions := []*LabelPrediction{}
	var pred_file *os.File
	if pred_path != ""{
		pred_file, _ = os.Create(pred_path)
	}
	for _,sample := range test_dataset.Samples {
		prediction := classifier.Predict(sample)
		if pred_file != nil{
			pred_file.WriteString(strconv.FormatFloat(prediction, 'g', 5, 64) + "\n")
		}
		predictions = append(predictions, &(LabelPrediction{Label: sample.Label, Prediction: prediction}))
	}
	if pred_path != ""{
		defer pred_file.Close()
	}
		
	auc := AUC(predictions)
	return auc, predictions
}