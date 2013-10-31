package main

import (
	"dataworm"
	"flag"
	"os"
	"strconv"
	"sync"
)

/* Data Worm Data Analyze */

func main(){
	data_path := flag.String("input", "", "Data set path")
	output_path := flag.String("output", "feature_analyze.tsv", "Feature analyze results path")
	flag.Parse()
	
	dataset := dataworm.NewDataSet()
	var info_value map[int64]float64
	var err error
	var wait sync.WaitGroup
	wait.Add(2)
	go func(){
		dataset.Load(*data_path, -1, 1)
		wait.Done()
	}()
	
	go func(){
		info_value = dataworm.InformationValue(dataset)
		wait.Done()
	}()
	
	wait.Wait()
	
	out, err := os.Create(*output_path)
	if err != nil{
		return
	}
	defer out.Close()
	
	for fid, iv := range info_value{
		out.WriteString(strconv.FormatInt(fid, 10) + "\t" + strconv.FormatFloat(iv, 'g', 10, 64) + "\n")
	} 
}