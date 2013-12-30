package core

import (
	"math"
	"math/rand"
)

func XORDataSet(n int) *DataSet{
	ret := NewDataSet()
	for i := 0; i < n; i++ {
		x := 2 * (float64(rand.Intn(2)) - 0.5)
		y := 2 * (float64(rand.Intn(2)) - 0.5)

		label := 1

		if x * y < 0.0 {
			label = 0
		}

		sample := NewSample()
		sample.Label = label
		sample.AddFeature(Feature{Id: 1, Value: x})
		sample.AddFeature(Feature{Id: 2, Value: y})
		sample.AddFeature(Feature{Id: 3, Value: 1.0})
		ret.AddSample(sample)
	}
	return ret
}

func LinearDataSet(n int) *DataSet {
	ret := NewDataSet()
	for i := 0; i < n; i++{
		sample := NewSample()
		sample.Label = 0
		for f := 0; f < 100; f++{
			if rand.Intn(10) != 1 {
				continue
			}
			if f < 20 {
				sample.Label += 1
			} else if f > 80 {
				sample.Label -= 1
			}
			sample.AddFeature(Feature{Id: int64(f), Value: 1.0})
		}
		if sample.Label > 0 {
			sample.Label = 1
		} else {
			sample.Label = 0
		}
		ret.AddSample(sample)
	}
	return ret
}

func SinusoidalDataSet(n int) *RealDataSet {
	ret := NewRealDataSet()

	min := -5.0
	max := 5.0
	amp := 1.0
	noise := 0.05
	period := 4.0
	interval := (max - min) / float64(n)
	for i := 0; i < n; i++ {
		x := min + interval * float64(i) + 0.5*interval
		y := math.Sin((x-min)*2*math.Pi/period) * amp + rand.NormFloat64()*noise
		sample := NewRealSample()
		sample.AddFeature(Feature{Id: int64(1), Value: x})
		sample.Value = y
		ret.AddSample(sample)
	}

	return ret
}
