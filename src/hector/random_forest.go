package hector

import (
	"math/rand"
	"strconv"
	"os"
	"bufio"
	"fmt"
)

type RandomForestParams struct {
	TreeCount int
	FeatureCount float64
}

type RandomForest struct {
	trees []Tree
	params RandomForestParams
	cart CART
	continuous_features bool
}

func (self *RandomForest) SaveModel(path string){
	file, _ := os.Create(path)
	defer file.Close()
	for _, tree := range self.trees {
		buf := tree.ToString()
		file.Write(buf)
		file.WriteString("\n#\n")
	}
}

func (self *RandomForest) LoadModel(path string){
	file, _ := os.Open(path)
	defer file.Close()

	self.trees = []Tree{}
	scanner := bufio.NewScanner(file)
	text := ""
	for scanner.Scan() {
		line := scanner.Text()
		if line == "#" {
			tree := Tree{}
			tree.FromString(text)
			self.trees = append(self.trees, tree)
			text = ""
		} else {
			text += line + "\n"
		}
	}
}

func (dt *RandomForest) Init(params map[string]string){
	dt.trees = []Tree{}
	dt.cart.Init(params)
	tree_count,_ := strconv.ParseInt(params["tree-count"], 10, 64)
	feature_count,_ := strconv.ParseInt(params["feature-count"], 10, 64)
	dt.params.TreeCount = int(tree_count)
	dt.params.FeatureCount = float64(feature_count)	
}

func (dt *RandomForest) Train(dataset * DataSet) {
	samples := []*MapBasedSample{}
	featureset := make(map[int64]bool)
	feature_weights := make(map[int64]float64)
	for _, sample := range dataset.Samples{
		if !dt.continuous_features {
			for _, f := range sample.Features {
				_, ok := feature_weights[f.Id]
				if !ok {
					feature_weights[f.Id] = f.Value
				}
				if feature_weights[f.Id] != f.Value {
					dt.continuous_features = true
				}
			}
		}
		msample := sample.ToMapBasedSample()
		samples = append(samples, msample)
		for fid, _ := range msample.Features {
			featureset[fid] = true
		}
	}
	dt.cart.continuous_features = dt.continuous_features
	
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
		fmt.Printf(".")
	}
	fmt.Println()
}

func (dt *RandomForest) Predict(sample * Sample) float64 {
	msample := sample.ToMapBasedSample()
	predictions := 0.0
	total := 0.0
	for _, tree := range dt.trees{
		node, _ := PredictBySingleTree(&tree, msample)
		predictions += node.prediction.GetValue(1)
		total += 1.0
	}
	return predictions / total
}

func (dt *RandomForest) PredictMultiClass(sample * Sample) *ArrayVector {
	msample := sample.ToMapBasedSample()
	predictions := NewArrayVector()
	total := 0.0
	for _, tree := range dt.trees{
		node, _ := PredictBySingleTree(&tree, msample)
		predictions.AddVector(node.prediction, 1.0)
		total += 1.0
	}
	predictions.Scale(1.0 / total)
	return predictions
}
