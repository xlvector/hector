package hector

type Classifier interface {
	Init(params map[string]string)
	Train(dataset * DataSet)
	Predict(sample * Sample) float64
}