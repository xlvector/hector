package hector

import (
	"math/rand"
)

func XORDataSet(n int) *DataSet{
	ret := NewDataSet()
	for i := 0; i < n; i++ {
		x := 2 * (float64(rand.Intn(2)) - 0.5)
		y := 2 * (float64(rand.Intn(2)) - 0.5)

		label := 1.0

		if x * y < 0.0 {
			label = 0.0
		}

		sample := NewSample()
		sample.Label = label
		sample.AddFeature(Feature{Id: 1, Value: x})
		sample.AddFeature(Feature{Id: 2, Value: y})
		ret.AddSample(sample)
	}
	return ret
}

func LinearDataSet(n int) *DataSet {
	ret := NewDataSet()
	for i := 0; i < n; i++{
		sample := NewSample()
		sample.Label = 0.0
		for f := 0; f < 100; f++{
			if rand.Intn(10) != 1 {
				continue
			}
			if f < 20 {
				sample.Label += 1.0
			} else if f > 80 {
				sample.Label -= 1.0
			}
			sample.AddFeature(Feature{Id: int64(f), Value: 1.0})
		}
		if sample.Label > 0.0 {
			sample.Label = 1.0
		} else {
			sample.Label = 0.0
		}
		ret.AddSample(sample)
	}
	return ret
}