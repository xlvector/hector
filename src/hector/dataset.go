package hector

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"sort"
)

type CombinedFeature []string

type FeatureSplit []float64

func FindCategory(split []float64, value float64) int {
	return sort.Search(len(split), func(i int) bool { return split[i] >= value })
}

type RawDataSet struct {
	Samples []*RawSample
	FeatureKeys map[string]bool
}

func NewRawDataSet() *RawDataSet{
	ret := RawDataSet{}
	ret.Samples = []*RawSample{}
	ret.FeatureKeys = make(map[string]bool)
	return &ret
}

func (d *RawDataSet) AddSample(sample *RawSample) {
	d.Samples = append(d.Samples, sample)
}

func (d *RawDataSet) ToDataSet(splits map[string][]float64, combinations []CombinedFeature) *DataSet{
	out_data := NewDataSet()
	for _, sample := range d.Samples {
		out_sample := NewSample()
		out_sample.Label = sample.Label
		if splits != nil{
			for fkey_str, fvalue_str := range sample.Features {
				fkey := ""
				fvalue := 0.0
				if GetFeatureType(fkey_str) == FeatureTypeEnum.CONTINUOUS_FEATURE {
					split, ok := splits[fkey_str]
					if ok {
						cat := FindCategory(split, ParseFloat64(fvalue_str))
						fkey = fkey_str + "_" + strconv.FormatInt(int64(cat), 10)
						fvalue = 1.0
					} else {
						fvalue = ParseFloat64(fvalue_str)
					}
					out_sample.AddFeature(Feature{Id:Hash(fkey), Value:fvalue})
				}
			}
		}
		for _, combination := range combinations {
			fkey := ""
			for _, ckey := range combination {
				fkey += ckey
				fkey += ":"
				fkey += sample.GetFeatureValue(ckey)
				fkey += "_"
			}
			out_sample.AddFeature(Feature{Id:Hash(fkey), Value:1.0})
		}
		out_data.AddSample(out_sample)
	}
	return out_data
}

func (d *RawDataSet) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.Replace(scanner.Text(), " ", "\t", -1)
		tks := strings.Split(line, "\t")
		sample := NewRawSample()
		for i, tk := range tks {
			if i == 0 {
				label, err := strconv.ParseInt(tk, 10, 16)
				if err != nil {
					break
				}
				if label > 0 {
					sample.Label = 1.0
				} else {
					sample.Label = 0.0
				}
			} else {
				kv := strings.Split(tk, ":")
				sample.Features[kv[0]] = kv[1]
				d.FeatureKeys[kv[0]] = true
			}
		}
		d.AddSample(sample)
	}
	if scanner.Err() != nil {
		return scanner.Err()
	}
	return nil
}


type DataSet struct {
	Samples []*Sample
}

func NewDataSet() *DataSet {
	ret := DataSet{}
	ret.Samples = []*Sample{}
	return &ret
}

func (d *DataSet) AddSample(sample *Sample) {
	d.Samples = append(d.Samples, sample)
}

func (d *DataSet) Load(path string, global_bias_feature_id int64) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.Replace(scanner.Text(), " ", "\t", -1)
		tks := strings.Split(line, "\t")
		sample := Sample{Features: []Feature{}, Label: 0}
		for i, tk := range tks {
			if i == 0 {
				label, _ := strconv.Atoi(tk)
				sample.Label = label
			} else {
				kv := strings.Split(tk, ":")
				feature_id, err := strconv.ParseInt(kv[0], 10, 64)
				if err != nil {
					break
				}
				feature_value := 1.0
				if len(kv) > 1 {
					feature_value, err = strconv.ParseFloat(kv[1], 64)
					if err != nil {
						break
					}
				}
				feature := Feature{feature_id, feature_value}
				sample.Features = append(sample.Features, feature)
			}
		}
		if global_bias_feature_id >= 0 {
			sample.Features = append(sample.Features, Feature{global_bias_feature_id, 1.0})
		}
		d.AddSample(&sample)
	}
	if scanner.Err() != nil {
		return scanner.Err()
	}
	return nil
}

func RemoveLowFreqFeatures(dataset *DataSet, threshold float64) {
	freq := NewVector()

	for _, sample := range dataset.Samples {
		for _, feature := range sample.Features{
			freq.AddValue(feature.Id, 1.0)
		}
	}

	for _, sample := range dataset.Samples {
		features := []Feature{}
		for _, feature := range sample.Features{
			if freq.GetValue(feature.Id) > threshold {
				features = append(features, feature)
			}
		}
		sample.Features = features
	}
}

func (d *DataSet) Split(f func(int) bool) *DataSet {
	out_data := NewDataSet()
	for i, sample := range d.Samples {
		if f(i){
			out_data.AddSample(sample)
		}
	}
	return out_data
}
