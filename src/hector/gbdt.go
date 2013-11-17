package hector

import (
	"strconv"
	"math"
	"fmt"
)

type GBDT struct {
	dts []*RegressionTree
	tree_count int
	shrink float64
}


func (self *GBDT) SaveModel(path string){

}

func (self *GBDT) LoadModel(path string){
	
}

func (c *GBDT) Init(params map[string]string) {
	tree_count,_ := strconv.ParseInt(params["tree-count"], 10, 64)
	c.tree_count = int(tree_count)
	for i := 0; i < c.tree_count; i++{
		dt := RegressionTree{}
		dt.Init(params)
		c.dts = append(c.dts, &dt)
	}
	c.shrink, _ = strconv.ParseFloat(params["learning-rate"], 64)
}

func (c *GBDT) RMSE(dataset *DataSet) float64 {
	rmse := 0.0
	n := 0.0
	for _, sample := range dataset.Samples {
		rmse += (sample.Label) * (sample.Label)
		n += 1.0
	}
	return math.Sqrt(rmse / n)
}

func (c *GBDT) Train(dataset *DataSet){
	for k, dt := range c.dts {
		dt.Train(dataset)
		for _, sample := range dataset.Samples {
			sample.Label -= c.shrink * dt.Predict(sample)
		}
		if k % 10 == 0 {
			fmt.Println(c.RMSE(dataset))
		}
	}
}

func (c *GBDT) Predict(sample *Sample) float64 {
	ret := 0.0
	for _, dt := range c.dts {
		ret += c.shrink * dt.Predict(sample)
	}
	return ret
}

