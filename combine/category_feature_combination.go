package combine

import (
	"fmt"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/eval"
	"github.com/xlvector/hector/lr"
	"math/rand"
)

type CategoryFeatureCombination struct {
	algo                 *lr.EPLogisticRegression
	feature_combinations []core.CombinedFeature
	output               string
}

func (c *CategoryFeatureCombination) Init(params map[string]string) {
	c.algo = &(lr.EPLogisticRegression{})
	c.algo.Init(params)
	c.output = params["output"]
}

func (c *CategoryFeatureCombination) OneCVAUC(dataset0 *core.RawDataSet, combines []core.CombinedFeature, total_cv, cv int) float64 {
	dataset := dataset0.ToDataSet(nil, combines)

	train := dataset.Split(func(i int) bool { return i%total_cv != cv })

	c.algo.Train(train)

	test := dataset.Split(func(i int) bool { return i%total_cv == cv })

	predictions := []*eval.LabelPrediction{}
	for _, sample := range test.Samples {
		pred := c.algo.Predict(sample)
		lp := eval.LabelPrediction{Label: sample.Label, Prediction: pred}
		predictions = append(predictions, &lp)
	}
	auc := eval.AUC(predictions)
	c.algo.Clear()
	return auc
}

func (c *CategoryFeatureCombination) FindCombination(dataset *core.RawDataSet) []core.CombinedFeature {
	features := []string{}
	for fkey, _ := range dataset.FeatureKeys {
		features = append(features, fkey)
	}
	candidate_column_combines := []core.CombinedFeature{}
	c.feature_combinations = []core.CombinedFeature{}

	for i, fi := range features {
		c.feature_combinations = append(c.feature_combinations, core.CombinedFeature{fi})
		for j, fj := range features[i+1:] {
			candidate_column_combines = append(candidate_column_combines, core.CombinedFeature{fi, fj})
			for k, fk := range features[i+j+1:] {
				candidate_column_combines = append(candidate_column_combines, core.CombinedFeature{fi, fj, fk})
				for _, ft := range features[i+j+k+1:] {
					candidate_column_combines = append(candidate_column_combines, core.CombinedFeature{fi, fj, fk, ft})
				}
			}
		}
	}
	fmt.Printf("candidates %d\n", len(candidate_column_combines))
	used_combines := make(map[int]bool)

	total_cv := 3

	best_auc := 0.0
	best_combines := -1
	for {
		if len(used_combines) == len(candidate_column_combines) {
			break
		}
		ok := false
		for i, column_combines := range candidate_column_combines {
			_, used := used_combines[i]
			if used {
				continue
			}
			temp_combines := c.feature_combinations
			temp_combines = append(temp_combines, column_combines)

			ave_auc := 0.0
			for cv := 0; cv < total_cv; cv++ {
				ave_auc += c.OneCVAUC(dataset, temp_combines, total_cv, cv)
			}
			ave_auc /= float64(total_cv)
			if best_auc < ave_auc {
				best_auc = ave_auc
				best_combines = i
				ok = true
				if rand.Intn(10) == 1 {
					break
				}
			}
		}
		if !ok {
			break
		}
		used_combines[best_combines] = true
		c.feature_combinations = append(c.feature_combinations, candidate_column_combines[best_combines])
		fmt.Println(best_auc)
		fmt.Println(c.feature_combinations)
	}

	return c.feature_combinations
}
