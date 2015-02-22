package main

import (
	"flag"
	"fmt"
	"github.com/xlvector/hector/core"
	"sort"
)

type FeatureValue struct {
	Name  string
	Value float64
}

type FeatureValueList []FeatureValue

func (ms FeatureValueList) Len() int {
	return len(ms)
}

func (ms FeatureValueList) Less(i, j int) bool {
	return ms[i].Value > ms[j].Value
}

func (ms FeatureValueList) Swap(i, j int) {
	ms[i], ms[j] = ms[j], ms[i]
}

func main() {
	path := flag.String("input", "", "path of dataset")
	flag.Parse()

	ds := core.NewDataSet()
	ds.Load(*path, -1)
	iv := core.InformationValue(ds)
	fs := make(FeatureValueList, 0, len(iv))
	for f, v := range iv {
		fs = append(fs, FeatureValue{Name: ds.FeatureNameIdMap[f], Value: v})
	}
	sort.Sort(fs)
	for _, f := range fs {
		fmt.Printf("%s\t%v\n", f.Name, f.Value)
	}
}
