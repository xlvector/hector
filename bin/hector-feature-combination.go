package main

import(
	"hector"
	"strings"
	"os"
)

func main(){
	train, _, _, _, params := hector.PrepareParams()
	
	feature_combination := hector.CategoryFeatureCombination{}
	feature_combination.Init(params)

	dataset := hector.NewRawDataSet()
	dataset.Load(train)

	combinations := feature_combination.FindCombination(dataset)

	output := params["output"]

	file, _ := os.Create(output)
	defer file.Close()

	for _, combination := range combinations {
		file.WriteString(strings.Join(combination, "\t") + "\n")
	}
}