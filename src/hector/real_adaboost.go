package hector

import (
	"math/rand"
	"math"
)

type RealAdaboost struct {
	trees []Tree
	cart CART
}

func NormArray(v []float64){
	sum := 0.0
	for _, e := range v{
		sum += e
	}
	for i, e := range v{
		v[i] = e / sum
	}
}

func SampleIndex(v []float64) int{
	p := rand.Float64()
	for i, e := range v{
		p -= e
		if p <= 0{
			return i
		}
	}
	return len(v) - 1
}

func (c *RealAdaboost) Init(params map[string]string){
	c.cart.Init(params)
}

func (c *RealAdaboost) Train(dataset DataSet){
	samples := []*MapBasedSample{}
	w := []float64{}
	for sample := range dataset.Samples{
		samples = append(samples, sample.ToMapBasedSample())
		w = append(w, 1.0)
	}
	
	for m := 0; m < 2; m++ {
		NormArray(w)
		sub_samples := []*MapBasedSample{}
		for k:= 0; k < len(samples); k++{
			rk := SampleIndex(w)
			sub_samples = append(sub_samples, samples[rk])	
		}
		tree := c.cart.SingleTreeBuild(sub_samples, nil)
		c.trees = append(c.trees, tree)
		for k, sample := range samples {
			node := c.cart.PredictBySingleTree(&tree, sample)
			pred := node.prediction * 0.999 + 0.001
			alpha := 0.5 * math.Log(pred / (1.0 - pred))
			
			if sample.Label > 0.0{
				w[k] *= math.Exp(-1 * alpha)
			} else {
				w[k] *= math.Exp(alpha)
			}
		}
	}
}

func (c *RealAdaboost) Predict(sample Sample) float64 {
	msample := sample.ToMapBasedSample()
	total := 0.0
	for _, tree := range c.trees {
		node := c.cart.PredictBySingleTree(&tree, msample)
		pred := node.prediction * 0.999 + 0.001
		total += 0.5 * math.Log(pred / (1.0 - pred))
	}
	return Sigmoid(total)
}