package svm

import (
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/lr"
	"math"
	"math/rand"
	"strconv"
)

func Distance(x, y *core.Vector) float64 {
	z := x.Copy()
	z.AddVector(y, -1)
	d := z.NormL2()
	return d
}

func RBFKernel(x, y *core.Vector, radius float64) float64 {
	d := Distance(x, y)
	ret := math.Exp(-1.0 * d / radius)
	return ret
}

type L1VM struct {
	sv     []*core.Vector
	ftrl   *lr.FTRLLogisticRegression
	radius float64
	count  int
}

func (self *L1VM) SaveModel(path string) {

}

func (self *L1VM) LoadModel(path string) {

}

func (c *L1VM) Init(params map[string]string) {
	c.ftrl = &(lr.FTRLLogisticRegression{})
	c.ftrl.Init(params)
	c.radius, _ = strconv.ParseFloat(params["radius"], 64)
	count, _ := strconv.ParseInt(params["sv"], 10, 64)
	c.count = int(count)
}

func (c *L1VM) Predict(sample *core.Sample) float64 {
	x := sample.GetFeatureVector()
	return c.PredictVector(x)
}

func (c *L1VM) PredictVector(x *core.Vector) float64 {
	s := core.NewSample()
	for k, xs := range c.sv {

		s.AddFeature(core.Feature{Id: int64(k), Value: RBFKernel(xs, x, c.radius)})
	}
	return c.ftrl.Predict(s)
}

func (c *L1VM) Train(dataset *core.DataSet) {
	c.sv = []*core.Vector{}
	kernel_dataset := core.NewDataSet()

	positive := []int{}
	negative := []int{}
	for i, si := range dataset.Samples {
		if si.Label > 0.0 {
			positive = append(positive, i)
		} else {
			negative = append(negative, i)
		}
	}

	perm_positive := rand.Perm(len(positive))

	for i, k := range perm_positive {
		if i > c.count {
			break
		}
		c.sv = append(c.sv, dataset.Samples[positive[k]].GetFeatureVector())
	}

	perm_negative := rand.Perm(len(negative))

	for i, k := range perm_negative {
		if i > c.count {
			break
		}
		c.sv = append(c.sv, dataset.Samples[negative[k]].GetFeatureVector())
	}

	for _, si := range dataset.Samples {
		xi := si.GetFeatureVector()
		tsample := core.NewSample()
		tsample.Label = si.Label
		for j, xj := range c.sv {
			tsample.AddFeature(core.Feature{Id: int64(j), Value: RBFKernel(xi, xj, c.radius)})
		}
		kernel_dataset.AddSample(tsample)
	}

	c.ftrl.Train(kernel_dataset)
}
