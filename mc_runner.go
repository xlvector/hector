/*
Package hector is a golang based machine learning lib. It intend to implement all famous machine learning algoirhtms by golang.
Currently, it only support algorithms which can solve binary classification problems. Supported algorithms include:
1. Decision Tree (CART, Random Forest, GBDT)
2. Logistic Regression
3. SVM
4. Neural Network
*/
package hector

import (
	"github.com/xlvector/hector/algo"
	"github.com/xlvector/hector/core"
	"os"
	"strconv"
)

func MultiClassRun(classifier algo.MultiClassClassifier, train_path string, test_path string, pred_path string, params map[string]string) (float64, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := core.NewDataSet()

	err := train_dataset.Load(train_path, global)

	if err != nil {
		return 0.5, err
	}

	test_dataset := core.NewDataSet()
	err = test_dataset.Load(test_path, global)
	if err != nil {
		return 0.5, err
	}
	classifier.Init(params)
	accuracy := MultiClassRunOnDataSet(classifier, train_dataset, test_dataset, pred_path, params)

	return accuracy, nil
}

func MultiClassTrain(classifier algo.MultiClassClassifier, train_path string, params map[string]string) error {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := core.NewDataSet()

	err := train_dataset.Load(train_path, global)

	if err != nil {
		return err
	}

	classifier.Init(params)
	classifier.Train(train_dataset)

	model_path, _ := params["model"]

	if model_path != "" {
		classifier.SaveModel(model_path)
	}

	return nil
}

func MultiClassTest(classifier algo.MultiClassClassifier, test_path string, pred_path string, params map[string]string) (float64, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)

	model_path, _ := params["model"]
	classifier.Init(params)
	if model_path != "" {
		classifier.LoadModel(model_path)
	} else {
		return 0.0, nil
	}

	test_dataset := core.NewDataSet()
	err := test_dataset.Load(test_path, global)
	if err != nil {
		return 0.0, err
	}

	accuracy := MultiClassRunOnDataSet(classifier, nil, test_dataset, pred_path, params)

	return accuracy, nil
}

func MultiClassRunOnDataSet(classifier algo.MultiClassClassifier, train_dataset, test_dataset *core.DataSet, pred_path string, params map[string]string) float64 {

	if train_dataset != nil {
		classifier.Train(train_dataset)
	}

	var pred_file *os.File
	if pred_path != "" {
		pred_file, _ = os.Create(pred_path)
	}
	accuracy := 0.0
	total := 0.0
	for _, sample := range test_dataset.Samples {
		prediction := classifier.PredictMultiClass(sample)
		label, _ := prediction.KeyWithMaxValue()
		if int(label) == sample.Label {
			accuracy += 1.0
		}
		total += 1.0
		if pred_file != nil {
			pred_file.WriteString(strconv.Itoa(label) + "\n")
		}
	}
	if pred_path != "" {
		defer pred_file.Close()
	}

	return accuracy / total
}
