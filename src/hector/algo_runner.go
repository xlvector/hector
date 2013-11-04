package hector

import (
	"strconv"
	"os"
)

func AlgorithmRun(classifier Classifier, train_path string, test_path string, pred_path string, params map[string]string) (float64, []*LabelPrediction, error) {
	classifier.Init(params)
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := NewDataSet()

	err := train_dataset.Load(train_path, global)
	
	if err != nil{
		return 0.5, nil, err
	}
	
	classifier.Train(train_dataset)
	
	test_dataset := NewDataSet()
	err = test_dataset.Load(test_path, global)
	if err != nil{
		return 0.5, nil, err
	}
	
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
	return auc, predictions, nil
}