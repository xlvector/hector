package main

import (
	"bufio"
	"fmt"
	"github.com/xlvector/hector"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/eval"
	"github.com/xlvector/hector/lr"
	"os"
	"strconv"
	"sync"
)

func SplitFile(input string, total int, part int) (string, string, error) {
	file, err := os.Open(input)
	if err != nil {
		return "", "", err
	}
	defer file.Close()

	train_path := input + ".train." + strconv.Itoa(part)
	train_file, err := os.Create(train_path)
	if err != nil {
		return "", "", err
	}
	defer train_file.Close()

	test_path := input + ".test." + strconv.Itoa(part)
	test_file, err := os.Create(test_path)
	if err != nil {
		return "", "", err
	}
	defer test_file.Close()

	scanner := bufio.NewScanner(file)
	k := 0
	for scanner.Scan() {
		if k%total == part {
			test_file.WriteString(scanner.Text() + "\n")
		} else {
			train_file.WriteString(scanner.Text() + "\n")
		}
		k += 1
	}
	return train_path, test_path, nil
}

func main() {
	train_path, test_path, pred_path, _, params := hector.PrepareParams()
	total := 5
	methods := []string{"ftrl", "fm"}
	all_methods_predictions := [][]*eval.LabelPrediction{}
	all_methods_test_predictions := [][]*eval.LabelPrediction{}
	for _, method := range methods {
		fmt.Println(method)
		average_auc := 0.0
		all_predictions := []*eval.LabelPrediction{}
		for part := 0; part < total; part++ {
			train, test, _ := SplitFile(train_path, total, part)
			classifier := hector.GetClassifier(method)

			auc, predictions, _ := hector.AlgorithmRun(classifier, train, test, "", params)
			fmt.Println("AUC:")
			fmt.Println(auc)
			average_auc += auc
			os.Remove(train)
			os.Remove(test)
			classifier = nil
			for _, pred := range predictions {
				all_predictions = append(all_predictions, pred)
			}
		}
		all_methods_predictions = append(all_methods_predictions, all_predictions)
		fmt.Println(average_auc / float64(total))

		classifier := hector.GetClassifier(method)
		fmt.Println(test_path)
		_, test_predictions, _ := hector.AlgorithmRun(classifier, train_path, test_path, "", params)
		all_methods_test_predictions = append(all_methods_test_predictions, test_predictions)
	}

	var wait sync.WaitGroup
	wait.Add(2)
	dataset := core.NewDataSet()
	go func() {
		for i, _ := range all_methods_predictions[0] {
			sample := core.NewSample()
			sample.Label = all_methods_predictions[0][i].Label
			for j, _ := range all_methods_predictions {
				feature := core.Feature{Id: int64(j), Value: all_methods_predictions[j][i].Prediction}
				sample.AddFeature(feature)
			}
			dataset.Samples <- sample
		}
		close(dataset.Samples)
		wait.Done()
	}()

	ensembler := lr.LinearRegression{}
	go func() {
		ensembler.Init(params)
		ensembler.Train(dataset)
		wait.Done()
	}()
	wait.Wait()

	fmt.Println(ensembler.Model)

	wait.Add(2)
	test_dataset := hector.NewDataSet()
	go func() {
		for i, _ := range all_methods_test_predictions[0] {
			sample := hector.NewSample()
			sample.Label = all_methods_test_predictions[0][i].Prediction
			for j, _ := range all_methods_test_predictions {
				feature := hector.Feature{Id: int64(j), Value: all_methods_test_predictions[j][i].Prediction}
				sample.AddFeature(feature)
			}
			test_dataset.Samples <- sample
		}
		close(test_dataset.Samples)
		wait.Done()
	}()

	go func() {
		pred_file, _ := os.Create(test_path + ".out")
		for sample := range test_dataset.Samples {
			prediction := sample.Label //ensembler.Predict(sample)
			pred_file.WriteString(strconv.FormatFloat(prediction, 'g', 5, 64) + "\n")
		}
		defer pred_file.Close()
		wait.Done()
	}()
	wait.Wait()
}
