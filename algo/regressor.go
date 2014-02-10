package algo

import (
	"github.com/xlvector/hector/core"
)

type Regressor interface {

	//Set training parameters from parameter map
	Init(params map[string]string)

	//Train model on a given dataset
	Train(dataset *core.RealDataSet)

	//Predict the output of an input sample
	Predict(sample *core.RealSample) float64

	SaveModel(path string)
	LoadModel(path string)
}
