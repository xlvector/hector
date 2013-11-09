package hector

import (
	//"fmt"
	"math"
)

type L1VM struct {
	sv []*Vector
	ftrl *FTRLLogisticRegression
}

func (c *L1VM) Init(params map[string]string){
	c.ftrl = &(FTRLLogisticRegression{})
	c.ftrl.Init(params)
}

func (c *L1VM) Kernel(x, y *Vector) float64{
	/*
	z := *x
	z.AddVector(y, -1)
	d := z.NormL2()
	ret := math.Exp(-1.0 * d / 100.0)
	return ret
	*/
	return x.Dot(y) / math.Sqrt(x.NormL2() * y.NormL2())
}

func (c *L1VM) Predict(sample *Sample) float64 {
	x := sample.GetFeatureVector()
	return c.PredictVector(x)
}

func (c *L1VM) PredictVector(x *Vector) float64 {
	s := NewSample()
	for k, xs := range c.sv {
		s.AddFeature(Feature{Id: int64(k), Value: c.Kernel(xs, x)})
	}
	return c.ftrl.Predict(s)
}

func (c *L1VM) Train(dataset *DataSet) {
	c.sv = []*Vector{}
	kernel_dataset := NewDataSet()
	for _, si := range dataset.Samples {
		xi := si.GetFeatureVector()
		c.sv = append(c.sv, xi)
		tsample := NewSample()
		for j, sj := range dataset.Samples {
			//if i == j{
			//	continue
			//}
			xj := sj.GetFeatureVector()
			tsample.AddFeature(Feature{Id: int64(j), Value: c.Kernel(xi, xj)})
		}
		kernel_dataset.AddSample(tsample)
	}

	c.ftrl.Train(kernel_dataset)
}