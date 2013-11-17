package hector

import(
	"math/rand"
	"fmt"
)

type SAOptAUC struct {
	Model map[int64]float64
}

func (self *SAOptAUC) SaveModel(path string){

}

func (self *SAOptAUC) LoadModel(path string){
	
}

func (algo *SAOptAUC) Init(params map[string]string) {
	algo.Model = make(map[int64]float64)
}

func (algo *SAOptAUC) TrainAUC(samples []*Sample) float64 {
	predictions := []*LabelPrediction{}
	for _, sample := range samples {
		pred := algo.Predict(sample)
		predictions = append(predictions, &(LabelPrediction{Label: sample.Label, Prediction: pred}))
	}
	return AUC(predictions)
}

func (algo *SAOptAUC) Train(dataset * DataSet) {
	algo.Model = make(map[int64]float64)
	samples := []*Sample{}
	for _, sample := range dataset.Samples {
		for _, feature := range sample.Features {
			algo.Model[feature.Id] = 1.0 / float64(len(sample.Features))
		}
		samples = append(samples, sample)
	}
	
	features := []int64{}
	for fid, _ := range algo.Model {
		features = append(features, fid)
	}
	
	prev_auc := 0.5
	for i := 0; i < 5000; i++ {
		add := rand.Float64()
		fid := features[rand.Intn(len(features))]
		fweight := algo.Model[fid]
		algo.Model[fid] = add
		auc := algo.TrainAUC(samples)
		
		if i % 500 == 0{
			fmt.Println(prev_auc)
		}
		
		if prev_auc < auc {
			prev_auc = auc
		} else{
			algo.Model[fid] = fweight
		}
	}
	fmt.Println(algo.Model)
}

func (algo *SAOptAUC) Predict(sample * Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value * feature.Value	
		}
	}
	return ret
}
