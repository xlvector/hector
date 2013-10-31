package hector

import (
	"strconv"
	"bufio"
	"strings"
	"os"
)

type DataSet struct {
	Samples chan Sample
}

func NewDataSet() *DataSet {
	ret := DataSet{}
	ret.Samples = make(chan Sample)
	return &ret
}

func (d *DataSet) AddSample(sample Sample){
	d.Samples <- sample
}

func (d *DataSet) Load(path string, global_bias_feature_id int64, steps int) error {
	for step := 0; step < steps; step++ {
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
	
		scanner := bufio.NewScanner(file)
	
		for scanner.Scan() {
			line := strings.Replace(scanner.Text(), " ", "\t", -1)
			tks := strings.Split(line, "\t")
			sample := Sample{Features : []Feature{}, Label : 0}
			for i, tk := range tks {
				if i == 0 {
					label, err := strconv.ParseInt(tk, 10, 16)
					if err != nil {
						break
					}
					if label > 0 {
						sample.Label = 1
					} else{
						sample.Label = 0
					}	
				} else{
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
			if global_bias_feature_id >= 0{
				sample.Features = append(sample.Features, Feature{global_bias_feature_id, 1.0})
			}
			d.Samples <- sample	
		}
		if scanner.Err() != nil{
			return scanner.Err()
		}
	}
	close(d.Samples)
	return nil	
}