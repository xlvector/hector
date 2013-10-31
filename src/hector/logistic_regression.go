package hector

import(
	"strconv"
)

type LogisticRegressionParams struct {
	LearningRate float64
	Regularization float64
	Steps int64
	GlobalBiasFeatureId int64
}

type LogisticRegression struct {
	Model map[int64]float64
	Params LogisticRegressionParams
}

func (algo *LogisticRegression) Init(params map[string]string) {
	algo.Model = make(map[int64]float64)
	
	algo.Params.LearningRate, _ = strconv.ParseFloat(params["learning-rate"], 64)
	algo.Params.Regularization, _ = strconv.ParseFloat(params["regularization"], 64)
}

func (algo *LogisticRegression) Train(dataset DataSet) {
	algo.Model = make(map[int64]float64)
	for sample := range dataset.Samples {
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
}

func (algo *LogisticRegression) Predict(sample Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value * feature.Value	
		}
	}
	return Sigmoid(ret)
}
