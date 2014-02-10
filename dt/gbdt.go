package dt

import (
	"bufio"
	"fmt"
	"github.com/xlvector/hector/core"
	"math"
	"os"
	"strconv"
)

type GBDT struct {
	dts        []*RegressionTree
	tree_count int
	shrink     float64
}

func (self *GBDT) SaveModel(path string) {
	file, _ := os.Create(path)
	defer file.Close()
	for _, dt := range self.dts {
		buf := dt.tree.ToString()
		file.Write(buf)
		file.WriteString("\n#\n")
	}
}

func (self *GBDT) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	self.dts = []*RegressionTree{}
	scanner := bufio.NewScanner(file)
	text := ""
	for scanner.Scan() {
		line := scanner.Text()
		if line == "#" {
			tree := Tree{}
			tree.FromString(text)
			dt := RegressionTree{tree: tree}
			self.dts = append(self.dts, &dt)
			text = ""
		} else {
			text += line + "\n"
		}
	}
}

func (c *GBDT) Init(params map[string]string) {
	tree_count, _ := strconv.ParseInt(params["tree-count"], 10, 64)
	c.tree_count = int(tree_count)
	for i := 0; i < c.tree_count; i++ {
		dt := RegressionTree{}
		dt.Init(params)
		c.dts = append(c.dts, &dt)
	}
	c.shrink, _ = strconv.ParseFloat(params["learning-rate"], 64)
}

func (c *GBDT) RMSE(dataset *core.DataSet) float64 {
	rmse := 0.0
	n := 0.0
	for _, sample := range dataset.Samples {
		rmse += (sample.Prediction) * (sample.Prediction)
		n += 1.0
	}
	return math.Sqrt(rmse / n)
}

func (c *GBDT) Train(dataset *core.DataSet) {
	for _, sample := range dataset.Samples {
		sample.Prediction = sample.LabelDoubleValue()
	}
	for k, dt := range c.dts {
		dt.Train(dataset)
		for _, sample := range dataset.Samples {
			sample.Prediction -= c.shrink * dt.Predict(sample)
		}
		if k%10 == 0 {
			fmt.Println(c.RMSE(dataset))
		}
	}
}

func (c *GBDT) Predict(sample *core.Sample) float64 {
	ret := 0.0
	for _, dt := range c.dts {
		ret += c.shrink * dt.Predict(sample)
	}
	return ret
}
