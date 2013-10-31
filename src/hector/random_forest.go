package hector

import (
	"math/rand"
	"strconv"
)

type RandomForestParams struct {
	TreeCount int
	FeatureCount float64
}

type RandomForest struct {
	trees []Tree
	params RandomForestParams
	cart CART
}

func (dt *RandomForest) Init(params map[string]string){
	dt.trees = []Tree{}
	dt.cart.Init(params)
	tree_count,_ := strconv.ParseInt(params["tree-count"], 10, 64)
	feature_count,_ := strconv.ParseInt(params["feature-count"], 10, 64)
	dt.params.TreeCount = int(tree_count)
	dt.params.FeatureCount = float64(feature_count)	
}

func (dt *RandomForest) Train(dataset DataSet) {
	samples := []*MapBasedSample{}
	featureset := make(map[int64]bool)
	for sample := range dataset.Samples{
		msample := sample.ToMapBasedSample()
		samples = append(samples, msample)
		for fid, _ := range msample.Features {
			featureset[fid] = true
		}
	}
	
	features := []int64{}
	for fid, _ := range featureset{
		features = append(features, fid)
	}
	
	for i:= 0; i < dt.params.TreeCount; i++{
		select_samples := []*MapBasedSample{}
		for k := 0; k < len(samples); k++{
			rand_sample := samples[rand.Intn(len(samples))]
			select_samples = append(select_samples, rand_sample)	
		}
		var select_features map[int64]bool
		if dt.params.FeatureCount > 0 {
			select_features = make(map[int64]bool)
			feature_count := int(float64(len(features)) * dt.params.FeatureCount)
			for k := 0; k < feature_count; k++{
				select_features[features[rand.Intn(len(features))]] = true
			}
		} else{
			select_features = featureset
		}
		
		tree := dt.cart.SingleTreeBuild(select_samples, select_features)
		dt.trees = append(dt.trees, tree)
	}
}

func (dt *RandomForest) Predict(sample Sample) float64 {
	msample := sample.ToMapBasedSample()
	predictions := 0.0
	total := 0.0
	for _, tree := range dt.trees{
		node := dt.cart.PredictBySingleTree(&tree, msample)
		predictions += node.prediction
		total += 1.0
	}
	return predictions / total
}
