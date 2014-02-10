package fm

import (
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
	"strconv"
)

type FactorizeMachine struct {
	w      *core.Vector
	v      []*core.Vector
	params FactorizeMachineParams
}

type FactorizeMachineParams struct {
	LearningRate   float64
	Regularization float64
	FactorNumber   int
}

func (self *FactorizeMachine) SaveModel(path string) {

}

func (self *FactorizeMachine) LoadModel(path string) {

}

func (c *FactorizeMachine) Predict(sample *core.Sample) float64 {
	for _, f := range sample.Features {
		c.w.RandomInit(f.Id, 0.1)
		for k, _ := range c.v {
			c.v[k].RandomInit(f.Id, 0.1)
		}
	}
	ret := c.w.DotFeatures(sample.Features)
	for k, _ := range c.v {
		a := c.v[k].DotFeatures(sample.Features)
		b := 0.0
		for _, f := range sample.Features {
			vkf := c.v[k].GetValue(f.Id)
			b += f.Value * f.Value * vkf * vkf
		}
		ret += 0.5 * (a*a - b)
	}
	return util.Sigmoid(ret)
}

func (c *FactorizeMachine) Init(params map[string]string) {
	c.w = core.NewVector()
	factor_number, _ := strconv.ParseInt(params["factors"], 10, 64)
	c.params.FactorNumber = int(factor_number)
	c.params.LearningRate, _ = strconv.ParseFloat(params["learning-rate"], 64)
	c.params.Regularization, _ = strconv.ParseFloat(params["regularization"], 64)

	for i := 0; i < c.params.FactorNumber; i++ {
		c.v = append(c.v, core.NewVector())
	}
}

func (c *FactorizeMachine) Train(dataset *core.DataSet) {
	n := 0
	for _, sample := range dataset.Samples {
		n += 1
		if n%10000 == 0 {
			c.params.LearningRate *= 0.9
		}
		pred := c.Predict(sample)
		err := sample.LabelDoubleValue() - pred

		vx := []float64{}
		for _, vf := range c.v {
			vx = append(vx, vf.DotFeatures(sample.Features))
		}
		for _, f := range sample.Features {
			fweight := c.w.GetValue(f.Id)
			fweight += c.params.LearningRate * (err*f.Value - c.params.Regularization*fweight)
			c.w.SetValue(f.Id, fweight)

			for k, _ := range c.v {
				vkx := c.v[k].GetValue(f.Id)
				vkx += c.params.LearningRate * (err*(f.Value*vx[k]-f.Value*f.Value*vkx) - c.params.Regularization*vkx)
				c.v[k].SetValue(f.Id, vkx)
			}
		}
	}
}
