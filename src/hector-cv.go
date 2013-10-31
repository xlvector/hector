package main

import(
	"hector"
	"os"
	"strconv"
	"fmt"
	"bufio"
)

func SplitFile(input string, total int, part int) (string, string, error) {
	file, err := os.Open(input)
	if err != nil {
		return "", "", err
	}
	defer file.Close()
	
	train_path := input + ".train." + strconv.Itoa(part)
	train_file, err := os.Create(train_path)
	if err != nil{
		return "", "", err
	}
	defer train_file.Close()
	
	test_path := input + ".test." + strconv.Itoa(part)
	test_file, err := os.Create(test_path)
	if err != nil{
		return "", "", err
	}
	defer test_file.Close()
	
	scanner := bufio.NewScanner(file)
	k := 0
	for scanner.Scan() {
		if k % total == part{
			test_file.WriteString(scanner.Text() + "\n")
		} else {
			train_file.WriteString(scanner.Text() + "\n")
		}
		k += 1
	}
	return train_path, test_path, nil
}

func main(){
	train_path, _, method, params := hector.PrepareParams()
	total := 10
	
	average_auc := 0.0
	for part := 0; part < total; part++ {
		train, test, _ := SplitFile(train_path, total, part)
		var classifier hector.Classifier
		
		if method == "lr"{
			classifier = &(hector.LogisticRegression{})
		} else if method == "ftrl" {
			classifier = &(hector.FTRLLogisticRegression{})
		} else if method == "rdt" {
			classifier = &(hector.RandomDecisionTree{})
		} else if method == "cart" {
			classifier = &(hector.CART{})	
		} else if method == "rf" {
			classifier = &(hector.RandomForest{})	
		} else if method == "rvm" {
			classifier = &(hector.RVM{})	
		} else if method == "real-adaboost" {
			classifier = &(hector.RealAdaboost{})	
		} else if method == "fm" {
			classifier = &(hector.FactorizeMachine{})	
		} else {
			classifier = &(hector.LogisticRegression{})
		}
		
		auc, _, _ := hector.AlgorithmRun(classifier, train, test, params)
		fmt.Println("AUC:")
		fmt.Println(auc)
		average_auc += auc
		os.Remove(train)
		os.Remove(test)
		classifier = nil
	}
	fmt.Println(average_auc / float64(total))
}