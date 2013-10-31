package hector

import (
	"sort"
)

type LabelPrediction struct {
	Label float64
	Prediction float64
}

type By func(p1, p2 *LabelPrediction) bool

type labelPredictionSorter struct {
	predictions []LabelPrediction
	by      By
}

func (s *labelPredictionSorter) Len() int {
	return len(s.predictions)
}

func (s *labelPredictionSorter) Swap(i, j int) {
	s.predictions[i], s.predictions[j] = s.predictions[j], s.predictions[i]
}

func (s *labelPredictionSorter) Less(i, j int) bool {
	return s.by(&s.predictions[i], &s.predictions[j])
}

func (by By) Sort(predictions []LabelPrediction) {
	sorter := &labelPredictionSorter{
		predictions: predictions,
		by:      by,
	}
	sort.Sort(sorter)
}

func AUC(predictions []LabelPrediction) float64 {
	prediction := func(p1, p2 *LabelPrediction) bool {
		return p1.Prediction > p2.Prediction
	}
	
	By(prediction).Sort(predictions)
	
	pn := 0.0
	nn := float64(len(predictions))
	ret := 0.0
	count := nn
	for i, lp := range predictions{
		if lp.Label > 0 {
			pn += 1.0
			nn -= 1.0
			ret += float64(count) - float64(i)
		}
	}
	ret2 := pn * (pn + 1) / 2.0;
	if pn * nn == 0.0{
		return 0.5
	}
	return (ret - ret2) / (pn * nn)
}
