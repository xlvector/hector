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
	"github.com/xlvector/hector/eval"
	"os"
	"strconv"
)

func AlgorithmRun(classifier algo.Classifier,
	train_path string, test_path string, pred_path string,
	params map[string]string) (float64, []*eval.LabelPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := core.NewDataSet()

	err := train_dataset.Load(train_path, global)

	if err != nil {
		return 0.5, nil, err
	}

	test_dataset := core.NewDataSet()
	err = test_dataset.Load(test_path, global)
	if err != nil {
		return 0.5, nil, err
	}
	classifier.Init(params)
	auc, predictions := AlgorithmRunOnDataSet(classifier, train_dataset,
		test_dataset, pred_path, params)

	return auc, predictions, nil
}

func AlgorithmTrain(classifier algo.Classifier, train_path string,
	params map[string]string) error {
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

func AlgorithmTest(classifier algo.Classifier, test_path string, pred_path string, params map[string]string) (float64, []*eval.LabelPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)

	model_path, _ := params["model"]
	classifier.Init(params)
	if model_path != "" {
		classifier.LoadModel(model_path)
	} else {
		return 0.0, nil, nil
	}

	test_dataset := core.NewDataSet()
	err := test_dataset.Load(test_path, global)
	if err != nil {
		return 0.0, nil, err
	}

	auc, predictions := AlgorithmRunOnDataSet(classifier, nil, test_dataset, pred_path, params)

	return auc, predictions, nil
}

func AlgorithmRunOnDataSet(classifier algo.Classifier, train_dataset, test_dataset *core.DataSet, pred_path string, params map[string]string) (float64, []*eval.LabelPrediction) {

	if train_dataset != nil {
		classifier.Train(train_dataset)
	}

	predictions := []*eval.LabelPrediction{}
	var pred_file *os.File
	if pred_path != "" {
		pred_file, _ = os.Create(pred_path)
	}
	for _, sample := range test_dataset.Samples {
		prediction := classifier.Predict(sample)
		if pred_file != nil {
			pred_file.WriteString(strconv.FormatFloat(prediction, 'g', 5, 64) + "\n")
		}
		predictions = append(predictions, &(eval.LabelPrediction{Label: sample.Label, Prediction: prediction}))
	}
	if pred_path != "" {
		defer pred_file.Close()
	}

	auc := eval.AUC(predictions)
	return auc, predictions
}

/* Regression */
func RegAlgorithmRun(regressor algo.Regressor, train_path string, test_path string, pred_path string, params map[string]string) (float64, []*eval.RealPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := core.NewRealDataSet()

	err := train_dataset.Load(train_path, global)

	if err != nil {
		return 0.5, nil, err
	}

	test_dataset := core.NewRealDataSet()
	err = test_dataset.Load(test_path, global)
	if err != nil {
		return 0.5, nil, err
	}
	regressor.Init(params)
	rmse, predictions := RegAlgorithmRunOnDataSet(regressor, train_dataset, test_dataset, pred_path, params)

	return rmse, predictions, nil
}

func RegAlgorithmTrain(regressor algo.Regressor, train_path string, params map[string]string) error {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := core.NewRealDataSet()

	err := train_dataset.Load(train_path, global)

	if err != nil {
		return err
	}

	regressor.Init(params)
	regressor.Train(train_dataset)

	model_path, _ := params["model"]

	if model_path != "" {
		regressor.SaveModel(model_path)
	}

	return nil
}

func RegAlgorithmTest(regressor algo.Regressor, test_path string, pred_path string, params map[string]string) (float64, []*eval.RealPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)

	model_path, _ := params["model"]
	regressor.Init(params)
	if model_path != "" {
		regressor.LoadModel(model_path)
	} else {
		return 0.0, nil, nil
	}

	test_dataset := core.NewRealDataSet()
	err := test_dataset.Load(test_path, global)
	if err != nil {
		return 0.0, nil, err
	}

	rmse, predictions := RegAlgorithmRunOnDataSet(regressor, nil, test_dataset, pred_path, params)

	return rmse, predictions, nil
}

func RegAlgorithmRunOnDataSet(regressor algo.Regressor, train_dataset, test_dataset *core.RealDataSet, pred_path string, params map[string]string) (float64, []*eval.RealPrediction) {

	if train_dataset != nil {
		regressor.Train(train_dataset)
	}

	predictions := []*eval.RealPrediction{}
	var pred_file *os.File
	if pred_path != "" {
		pred_file, _ = os.Create(pred_path)
	}
	for _, sample := range test_dataset.Samples {
		prediction := regressor.Predict(sample)
		if pred_file != nil {
			pred_file.WriteString(strconv.FormatFloat(prediction, 'g', 5, 64) + "\n")
		}
		predictions = append(predictions, &eval.RealPrediction{Value: sample.Value, Prediction: prediction})
	}
	if pred_path != "" {
		defer pred_file.Close()
	}

	rmse := eval.RegRMSE(predictions)
	return rmse, predictions
}
