package hector

import (
	"math"
	"strconv"
	"os"
	"bufio"
	"strings"
)

type EPLogisticRegressionParams struct {
	init_var, beta float64
}

type EPLogisticRegression struct {
	Model map[int64]*Gaussian
	params EPLogisticRegressionParams
}

func (algo *EPLogisticRegression) SaveModel(path string) {
	sb := StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g.mean)
		sb.Write("\t")
		sb.Float(g.vari)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (algo *EPLogisticRegression) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		fid, _ := strconv.ParseInt(tks[0], 10, 64)
		mean, _ := strconv.ParseFloat(tks[1], 64)
		vari, _ := strconv.ParseFloat(tks[2], 64)
		g := Gaussian{mean: mean, vari: vari}
		algo.Model[fid] = &g
	}
}

func (algo *EPLogisticRegression) Predict(sample * Sample) float64 {
	s := Gaussian{mean: 0.0, vari: 0.0}
	for _, feature := range sample.Features {
		if feature.Value == 0.0{
			continue
		}
		wi, ok := algo.Model[feature.Id]
		if !ok {
			wi = &(Gaussian{mean: 0.0, vari: algo.params.init_var})
		}
		s.mean += feature.Value * wi.mean
		s.vari += feature.Value * feature.Value * wi.vari
	}

	t := s
	t.vari += algo.params.beta
	return t.Integral(t.mean / math.Sqrt(t.vari))
}

func (algo *EPLogisticRegression) Init(params map[string]string) {
	algo.Model = make(map[int64]*Gaussian)
	algo.params.beta, _ = strconv.ParseFloat(params["beta"], 64)
	algo.params.init_var = 1.0
}

func (algo *EPLogisticRegression) Clear(){
	algo.Model = nil
	algo.Model = make(map[int64]*Gaussian)
}

func (algo *EPLogisticRegression) Train(dataset *DataSet) {

	for _, sample := range dataset.Samples {
		s := Gaussian{mean: 0.0, vari: 0.0}
		for _, feature := range sample.Features {
			if feature.Value == 0.0{
				continue
			}
			wi, ok := algo.Model[feature.Id]
			if !ok {
				wi = &(Gaussian{mean: 0.0, vari: algo.params.init_var})
				algo.Model[feature.Id] = wi
			}
			s.mean += feature.Value * wi.mean
			s.vari += feature.Value * feature.Value * wi.vari
		}

		t := s
		t.vari += algo.params.beta

		t2 := Gaussian{mean:0.0, vari: 0.0}
		if sample.Label > 0.0 {
			t2.UpperTruncateGaussian(t.mean, t.vari, 0.0)
		} else {
			t2.LowerTruncateGaussian(t.mean, t.vari, 0.0)
		}
		t.MultGaussian(&t2)
		s2 := t
		s2.vari += algo.params.beta
		s0 := s
		s.MultGaussian(&s2)

		for _, feature := range sample.Features {
			if feature.Value == 0.0{
				continue
			}
			wi0 := Gaussian{mean:0.0, vari:algo.params.init_var}
			w2 := Gaussian{mean:0.0, vari:0.0}
			wi, _ := algo.Model[feature.Id]
			w2.mean = (s.mean - (s0.mean - wi.mean * feature.Value)) / feature.Value
			w2.vari = (s.vari + (s0.vari - wi.vari * feature.Value * feature.Value)) / (feature.Value * feature.Value)
			wi.MultGaussian(&w2)
			wi_vari := wi.vari
			wi_new_vari := wi_vari * wi0.vari / (0.99 * wi0.vari + 0.01 * wi.vari)
			wi.vari = wi_new_vari
			wi.mean = wi.vari * (0.99 * wi.mean / wi_vari + 0.01 * wi0.mean / wi.vari)
			if wi.vari < algo.params.init_var * 0.01 {
				wi.vari = algo.params.init_var * 0.01
			}
			algo.Model[feature.Id] = wi
		}
	}
}