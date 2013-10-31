package hector

import(
	"sort"
	"math"
)

type WeightLabel struct {
	weight float64
	label int
}

type FeatureLabelDistribution struct {
	weight_label []WeightLabel
}

func NewFeatureLabelDistribution() *FeatureLabelDistribution{
	ret := FeatureLabelDistribution{}
	ret.weight_label = []WeightLabel{}
	return &ret
}

func (f *FeatureLabelDistribution) AddWeightLabel(weight float64, label int){
	wl := WeightLabel{weight:weight, label:label}
	f.weight_label = append(f.weight_label, wl)
}

func (f *FeatureLabelDistribution) Len() int {
	return len(f.weight_label)
}

func (f *FeatureLabelDistribution) Swap(i, j int) {
	f.weight_label[i], f.weight_label[j] = f.weight_label[j], f.weight_label[i]
}

func (f *FeatureLabelDistribution) Less(i, j int) bool {
	return (f.weight_label[i].weight < f.weight_label[j].weight)
}

func (f *FeatureLabelDistribution) PositiveCount() int {
	ret := 0
	for _, e := range f.weight_label{
		ret += e.label
	}
	return ret
}

func Gini(pleft, tleft, pright, tright float64) float64 {
	if tleft == 0.0 || tright == 0.0{
		return 1.0
	}
	p11 := pleft / tleft
	g1 := 1 - p11 * p11 - (1 - p11) * (1 - p11)
	p21 := pright / tright
	g2 := 1 - p21 * p21 - (1 - p21) * (1 - p21)
	ret := tleft * g1 / (tleft + tright) + tright * g2 / (tleft + tright)
	return ret	
}

func (f *FeatureLabelDistribution) BestSplitByGini(total, positive int) (float64, float64) {
	pright := float64(f.PositiveCount())
	tright := float64(len(f.weight_label))
	pleft := float64(positive) - pright
	tleft := float64(total) - tright
	min_gini := Gini(pleft, tleft, pright, tright)
	split := f.weight_label[0].weight
	prev_weight := f.weight_label[0].weight
	for _, wl := range f.weight_label{
		if prev_weight != wl.weight{
			gini := Gini(pleft, tleft, pright, tright)
			if gini < min_gini{
				min_gini = gini
				split = wl.weight
			}	
		}
		prev_weight = wl.weight
		tleft += 1.0
		tright -= 1.0
		pleft += float64(wl.label)
		pright -= float64(wl.label)
	}
	return split, min_gini	
}

func (f *FeatureLabelDistribution) InformationValue(global_total, global_positive int) float64 {
	with_total := len(f.weight_label)
	with_positive := f.PositiveCount()
	
	positives := []int{}
	negatives := []int{}
	
	positives = append(positives, global_positive - with_positive)
	negatives = append(negatives, (global_total - global_positive) - (with_total - with_positive))
	
	sort.Sort(f)
	
	prev_c := -1
	pos := 0
	total := 0
	for i, e := range f.weight_label {
		c := int(200.0 * float64(i) / float64(with_total))
		if c != prev_c {
			if total > 0{
				positives = append(positives, pos)
				negatives = append(negatives, total - pos)
				pos = 0
				total = 0
			}	
		}
		prev_c = c
		pos += e.label
		total += 1
	}
	if total > 0{
		positives = append(positives, pos)
		negatives = append(negatives, total - pos)
	}
	
	sum_positive := 0
	sum_negative := 0
	for _, v := range positives{
		sum_positive += v
	}
	for _, v := range negatives{
		sum_negative += v
	}
	iv := 0.0
	for i := range positives{
		positive_ratio := float64(positives[i]) / float64(sum_positive)
		negative_ratio := float64(negatives[i]) / float64(sum_negative)
		iv += (positive_ratio - negative_ratio) * math.Log((0.00001 + positive_ratio) / (0.00001 + negative_ratio))
	}
	return iv
}

func InformationValue(dataset *DataSet) map[int64]float64 {
	feature_weight_labels := make(map[int64]*FeatureLabelDistribution)
	total := 0
	positive := 0
	for sample := range dataset.Samples {
		total += 1
		positive += int(sample.Label)
		for _, feature := range sample.Features {
			_, ok := feature_weight_labels[feature.Id]
			if !ok {
				feature_weight_labels[feature.Id] = NewFeatureLabelDistribution()
			}
			feature_weight_labels[feature.Id].AddWeightLabel(feature.Value, int(sample.Label))
		}
	}
	
	ret := make(map[int64]float64)
	
	for fid, distribution := range feature_weight_labels{
		ret[fid] = distribution.InformationValue(total, positive)
	}
	return ret
}