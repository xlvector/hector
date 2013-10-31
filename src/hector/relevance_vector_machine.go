package hector

import (
	"math/rand"
	"math"
	"sync"
)

type RVM struct {
	centers []*Vector
	lr FTRLLogisticRegression
}

func (rvm *RVM) Init(params map[string]string){
	rvm.centers = []*Vector{}
	rvm.lr.Init(params)
}

func (rvm *RVM) CreateNewSample(sample *Sample) *Sample{
	new_sample := NewSample()
	feature_vector := sample.GetFeatureVector()
	for i:= 0; i < len(rvm.centers); i++{
		feature_vector.AddVector(rvm.centers[i], -1.0)
		distance := feature_vector.NormL2()
		distance = math.Exp(-0.5 * distance / 10000000.0)
		
		feature := Feature{Id: int64(i), Value: distance}
		new_sample.AddFeature(feature)
	}
	new_sample.Label = sample.Label
	return new_sample	
}

func (rvm *RVM) Train(dataset * DataSet){
	samples := []*Sample{}
	for sample := range dataset.Samples{
		if rand.Float64() < 0.1{
			rvm.centers = append(rvm.centers, sample.GetFeatureVector())
		}
		samples = append(samples, sample)		
	}
	
	new_dataset := NewDataSet()
	new_samples := []*Sample{}
	for _, sample := range samples {
		new_samples = append(new_samples, rvm.CreateNewSample(sample))		
	}
	
	var wait sync.WaitGroup
	wait.Add(2)
	go func(){
		for _, sample := range new_samples{
			new_dataset.AddSample(sample)
		}
		close(new_dataset.Samples)
		wait.Done()
	}()
	
	go func(){
		rvm.lr.Train(new_dataset)
		wait.Done()
	}()
	wait.Wait()
}

func (rvm *RVM) Predict(sample * Sample) float64{
	nsample := rvm.CreateNewSample(sample)
	return rvm.lr.Predict(nsample)
}