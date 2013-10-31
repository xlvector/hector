package hector

import (
	"math"
	"strconv"
)

type FTRLLogisticRegressionParams struct {
	Alpha, Beta, Lambda1, Lambda2 float64
}

type FTRLFeatureWeight struct {
	ni, zi float64
}

func (w *FTRLFeatureWeight) Wi(p FTRLLogisticRegressionParams) float64 {
	wi := 0.0
	if math.Abs(w.zi) > p.Lambda1 {
		wi = (Signum(w.zi) * p.Lambda1 - w.zi) / (p.Lambda2 + (p.Beta + math.Sqrt(w.ni)) / p.Alpha);	
	}
	return wi
}

type FTRLLogisticRegression struct {
	Model map[int64]FTRLFeatureWeight
	Params FTRLLogisticRegressionParams
}

func (algo *FTRLLogisticRegression) Predict(sample Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value.Wi(algo.Params) * feature.Value	
		}
	}
	return Sigmoid(ret)
}

func (algo *FTRLLogisticRegression) Init(params map[string]string) {
	algo.Model = make(map[int64]FTRLFeatureWeight)
	algo.Params.Alpha, _ = strconv.ParseFloat(params["alpha"], 64)
	algo.Params.Lambda1, _ = strconv.ParseFloat(params["lambda1"], 64)
	algo.Params.Lambda2, _ = strconv.ParseFloat(params["lambda2"], 64)
	algo.Params.Beta, _ = strconv.ParseFloat(params["beta"], 64)
}

func (algo *FTRLLogisticRegression) Train(dataset DataSet) {
	for sample := range dataset.Samples {
		prediction := algo.Predict(sample)
		err := sample.LabelDoubleValue() - prediction
		for _, feature := range sample.Features {
			model_feature_value, ok := algo.Model[feature.Id]
			if !ok {
				model_feature_value = FTRLFeatureWeight{0.0, 0.0}
			}
			zi := model_feature_value.zi
			ni := model_feature_value.ni
			gi := -1 * err * feature.Value
			sigma := (math.Sqrt(ni + gi * gi) - math.Sqrt(ni)) / algo.Params.Alpha
			wi := model_feature_value.Wi(algo.Params)
			zi += gi - sigma * wi
			ni += gi * gi
			algo.Model[feature.Id] = FTRLFeatureWeight{zi: zi, ni: ni}
		}
	}
}