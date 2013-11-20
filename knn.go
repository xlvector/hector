package hector

import (
	"strconv"
	"math"
	"math/rand"
)

type KNN struct {
	sv []*Vector
	labels []int
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
	z := x.Copy()
	z.AddVector(y, -1.0)
	ret := math.Exp(-1.0 * z.NormL2() / 20.0)
	return ret
}

func (c *KNN) Predict(sample *Sample) float64 {
	ret := c.PredictMultiClass(sample)
	return ret.GetValue(1)
}

func (c *KNN) PredictMultiClass(sample *Sample) *ArrayVector {
	x := sample.GetFeatureVector()
	predictions := []*LabelPrediction{}
	for i, s := range c.sv {
		predictions = append(predictions, &(LabelPrediction{Label:c.labels[i], Prediction:c.Kernel(s, x)}))
	}
	
	compare := func(p1, p2 *LabelPrediction) bool {
		return p1.Prediction > p2.Prediction
	}
	
	By(compare).Sort(predictions)

	ret := NewArrayVector()
	for i, pred := range predictions {
		if i > c.k{
			break
		}
		ret.AddValue(pred.Label, 1.0)
	}
	return ret
}

func (c *KNN) Train(dataset *DataSet) {
	c.sv = []*Vector{}
	c.labels = []int{}
	for i := 0; i < 1000; i++ {
		k := rand.Intn(len(dataset.Samples))
		c.sv = append(c.sv, dataset.Samples[k].GetFeatureVector())
		c.labels = append(c.labels, dataset.Samples[k].Label)
	}
}