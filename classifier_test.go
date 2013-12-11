package hector

import (
	"testing"
)
func TestClassifiers(t *testing.T){
	train_dataset := LinearDataSet(1000)
	test_dataset := LinearDataSet(500)

	algos := []string{"ep", "fm", "ftrl", "lr", "linear_svm", "lr_owlqn"}

	params := make(map[string]string)
	params["beta"] = "1.0"
	params["steps"] = "10"
	params["lambda1"] = "0.1"
	params["lambda2"] = "1.0"
	params["alpha"] = "0.1"
	params["max-depth"] = "20"
	params["min-leaf-size"] = "5"
	params["tree-count"] = "10"
	params["learning-rate"] = "0.05"
	params["regularization"] = "0.0001"
	params["e"] = "0.1"
	params["c"] = "0.1"
	params["gini"] = "1.0"
	params["factors"] = "10"

	for _, algo := range algos {
		classifier := GetClassifier(algo)
		classifier.Init(params)
		auc, _ := AlgorithmRunOnDataSet(classifier, train_dataset, test_dataset, "", params)

		t.Logf("auc of %s in linear dataset is %f", algo, auc)
		if auc < 0.9 {
			t.Error("auc less than 0.9 in linear dataset")
		}
	}
}

func TestClassifiersOnXOR(t *testing.T) {
	algos := []string{"ann", "rf", "rdt", "knn"}

	params := make(map[string]string)
	params["steps"] = "30"
	params["max-depth"] = "10"
	params["min-leaf-size"] = "10"
	params["tree-count"] = "100"
	params["learning-rate"] = "0.1"
	params["learning-rate-discount"] = "1.0"
	params["regularization"] = "0.0001"
	params["gini"] = "1.0"
	params["hidden"] = "15"
	params["k"] = "10"
	params["feature-count"] = "1.0"
	params["dt-sample-ratio"] = "1.0"

	for _, algo := range algos {
		train_dataset := XORDataSet(1000)
		test_dataset := XORDataSet(500)
		classifier := GetClassifier(algo)
		classifier.Init(params)
		auc, _ := AlgorithmRunOnDataSet(classifier, train_dataset, test_dataset, "", params)

		t.Logf("auc of %s in xor dataset is %f", algo, auc)
		if auc < 0.9 {
			t.Error("auc less than 0.9 in xor dataset")
		}
	}
}
