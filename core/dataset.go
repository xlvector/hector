package core

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/xlvector/hector/util"
)

type CombinedFeature []string

type FeatureSplit []float64

func FindCategory(split []float64, value float64) int {
	return sort.Search(len(split), func(i int) bool { return split[i] >= value })
}

/* RawDataSet */
type RawDataSet struct {
	Samples     []*RawSample
	FeatureKeys map[string]bool
}

func NewRawDataSet() *RawDataSet {
	ret := RawDataSet{}
	ret.Samples = []*RawSample{}
	ret.FeatureKeys = make(map[string]bool)
	return &ret
}

func (d *RawDataSet) AddSample(sample *RawSample) {
	d.Samples = append(d.Samples, sample)
}

func (d *RawDataSet) ToDataSet(splits map[string][]float64, combinations []CombinedFeature) *DataSet {
	out_data := NewDataSet()
	fm := make(map[string]int64)
	for _, sample := range d.Samples {
		out_sample := NewSample()
		out_sample.Label = sample.Label
		if splits != nil {
			for fkey_str, fvalue_str := range sample.Features {
				fkey := ""
				fvalue := 0.0
				if GetFeatureType(fkey_str) == FeatureTypeEnum.CONTINUOUS_FEATURE {
					split, ok := splits[fkey_str]
					if ok {
						cat := FindCategory(split, util.ParseFloat64(fvalue_str))
						fkey = fkey_str + "_" + strconv.FormatInt(int64(cat), 10)
						fvalue = 1.0
					} else {
						fvalue = util.ParseFloat64(fvalue_str)
					}
					fm[fkey] = util.Hash(fkey)
					out_sample.AddFeature(Feature{Id: util.Hash(fkey), Value: fvalue})
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
			fm[fkey] = util.Hash(fkey)
			out_sample.AddFeature(Feature{Id: util.Hash(fkey), Value: 1.0})
		}
		out_data.AddSample(out_sample)
	}
	f, _ := os.Create("features.tsv")
	defer f.Close()
	w := bufio.NewWriter(f)
	for k, v := range fm {
		w.WriteString(k + "\t" + strconv.FormatInt(v, 10) + "\n")
	}

	return out_data
}

func (d *RawDataSet) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	ch := make(chan string, 1000)
	go func() {
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				break
			}
			ch <- line
		}
		close(ch)
	}()

	n := 0
	for line := range ch {
		n += 1
		if n%10000 == 0 {
			fmt.Println(n, len(ch))
		}
		line = strings.Replace(line, " ", "\t", -1)
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
	return nil
}

/*Streaming*/
type StreamingDataSet struct {
	Samples chan *Sample
}

func NewStreamingDataSet() *StreamingDataSet {
	return &StreamingDataSet{
		Samples: make(chan *Sample, 10000),
	}
}

func (d *StreamingDataSet) AddSample(sample *Sample) {
	d.Samples <- sample
}

func (d *StreamingDataSet) Load(path string, global_bias_feature_id int64) error {
	for step := 0; step < 2; step++ {
		file, err := os.Open(path)
		defer file.Close()
		if err != nil {
			log.Fatalln("load file fail: ", err)
		}
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				break
			}
			tks := strings.Split(strings.TrimSpace(line), "\t")
			sample := Sample{Features: make([]Feature, 0, 20), Label: 0}
			for i, tk := range tks {
				if i == 0 {
					label, _ := strconv.Atoi(tk)
					sample.Label = label
				} else {
					kv := strings.Split(tk, ":")
					feature_id, err := strconv.ParseInt(kv[0], 10, 64)
					if err != nil {
						log.Fatalln("wrong feature: ", kv[0])
					}
					feature_value := 1.0
					if len(kv) > 1 {
						feature_value, err = strconv.ParseFloat(kv[1], 64)
						if err != nil {
							log.Fatalln("wrong value: ", kv[1])
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
	}
	close(d.Samples)
	return nil
}

/* DataSet */
type DataSet struct {
	Samples          []*Sample
	FeatureNameIdMap map[int64]string
	max_label        int
}

func NewDataSet() *DataSet {
	ret := DataSet{}
	ret.Samples = []*Sample{}
	ret.FeatureNameIdMap = make(map[int64]string)
	return &ret
}

func (d *DataSet) AddSample(sample *Sample) {
	d.Samples = append(d.Samples, sample)
	if d.max_label < sample.Label {
		d.max_label = sample.Label
	}
}

func (d *DataSet) Load(path string, global_bias_feature_id int64) error {
	fm := make(map[string]int64)

	ch := make(chan string, 1000)
	go func() {
		file, err := os.Open(path)
		defer file.Close()
		defer close(ch)
		if err != nil {
			log.Println("load file fail: ", err)
			return
		}

		scanner := bufio.NewScanner(file)

		for scanner.Scan() {
			line := strings.Replace(scanner.Text(), " ", "\t", -1)
			ch <- line
		}
	}()

	for line := range ch {
		tks := strings.Split(line, "\t")
		sample := Sample{Features: make([]Feature, 0, 20), Label: 0}
		for i, tk := range tks {
			if i == 0 {
				label, _ := strconv.Atoi(tk)
				sample.Label = label
				if d.max_label < label {
					d.max_label = label
				}
			} else {
				kv := strings.Split(tk, ":")
				feature_id, err := strconv.ParseInt(kv[0], 10, 64)
				if err != nil {
					feature_id = util.Hash(kv[0])
					fm[kv[0]] = feature_id
				}
				d.FeatureNameIdMap[feature_id] = kv[0]
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
	f, _ := os.Create("features.tsv")
	defer f.Close()
	w := bufio.NewWriter(f)
	for k, v := range fm {
		w.WriteString(k + "\t" + strconv.FormatInt(v, 10) + "\n")
	}

	log.Println("dataset size : ", len(d.Samples))
	return nil
}

func RemoveLowFreqFeatures(dataset *DataSet, threshold float64) {
	freq := NewVector()

	for _, sample := range dataset.Samples {
		for _, feature := range sample.Features {
			freq.AddValue(feature.Id, 1.0)
		}
	}

	for _, sample := range dataset.Samples {
		features := []Feature{}
		for _, feature := range sample.Features {
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
		if f(i) {
			out_data.AddSample(sample)
		}
	}
	return out_data
}

/* Real valued DataSet */
type RealDataSet struct {
	Samples []*RealSample
}

func NewRealDataSet() *RealDataSet {
	ret := RealDataSet{}
	ret.Samples = []*RealSample{}
	return &ret
}

func (d *RealDataSet) AddSample(sample *RealSample) {
	d.Samples = append(d.Samples, sample)
}

func (d *RealDataSet) Load(path string, global_bias_feature_id int64) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.Replace(scanner.Text(), " ", "\t", -1)
		tks := strings.Split(line, "\t")
		sample := RealSample{Features: []Feature{}, Value: 0.0}
		for i, tk := range tks {
			if i == 0 {
				value := util.ParseFloat64(tk)
				sample.Value = value
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
