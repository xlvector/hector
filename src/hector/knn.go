package hector

import (
	"strconv"
	"math"
)

type KNN struct {
	sv []*Sample
	k int
}


func (self *KNN) SaveModel(path string){

}

func (self *KNN) LoadModel(path string){
	
}

func (c *KNN) Init(params map[string]string){
	K, _ := strconv.ParseInt(params["k"], 10, 64)
	c.k = int(K)
}

func (c *KNN) Kernel(x, y *Vector) float64{
	z := *x
	z.AddVector(y, -1.0)
	return math.Exp(-1.0 * z.NormL2() / 10.0)
}

func (c *KNN) Predict(sample *Sample) float64 {
	x := sample.GetFeatureVector()
	predictions := []*LabelPrediction{}
	for _, s := range c.sv {
		predictions = append(predictions, &(LabelPrediction{Label:s.LabelDoubleValue(), Prediction:c.Kernel(s.GetFeatureVector(), x)}))
	}
	
	compare := func(p1, p2 *LabelPrediction) bool {
		return p1.Prediction > p2.Prediction
	}
	
	By(compare).Sort(predictions)

	total := 0.0
	positive := 0.0
	for i, pred := range predictions {
		if i > c.k{
			break
		}
		total += 1.0
		positive += pred.Label
	}
	return positive / total
}

func (c *KNN) Train(dataset *DataSet) {
	c.sv = dataset.Samples
}