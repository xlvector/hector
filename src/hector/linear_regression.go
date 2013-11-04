package hector

import(
	"strconv"
)

type LinearRegression struct {
	Model map[int64]float64
	Params LogisticRegressionParams
}

func (algo *LinearRegression) Init(params map[string]string) {
	algo.Model = make(map[int64]float64)
	
	algo.Params.LearningRate, _ = strconv.ParseFloat(params["learning-rate"], 64)
	algo.Params.Regularization, _ = strconv.ParseFloat(params["regularization"], 64)
}

func (algo *LinearRegression) Train(dataset * DataSet) {
	algo.Model = make(map[int64]float64)
	for step := 0; step < algo.Params.Steps; step++{
		for _, sample := range dataset.Samples {
			prediction := algo.Predict(sample)
			err := sample.LabelDoubleValue() - prediction
			for _, feature := range sample.Features {
				model_feature_value, ok := algo.Model[feature.Id]
				if !ok {
					model_feature_value = 0.0
				}
				model_feature_value += algo.Params.LearningRate * (err * feature.Value - algo.Params.Regularization * model_feature_value)
				algo.Model[feature.Id] = model_feature_value
			}
		}
		algo.Params.LearningRate *= 0.9
	}
}

func (algo *LinearRegression) Predict(sample * Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value * feature.Value	
		}
	}
	return ret
}
