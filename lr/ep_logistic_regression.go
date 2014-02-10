package lr

import (
	"bufio"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
	"math"
	"os"
	"strconv"
	"strings"
)

type EPLogisticRegressionParams struct {
	init_var, beta float64
}

type EPLogisticRegression struct {
	Model  map[int64]*util.Gaussian
	params EPLogisticRegressionParams
}

func (algo *EPLogisticRegression) SaveModel(path string) {
	sb := util.StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g.Mean)
		sb.Write("\t")
		sb.Float(g.Vari)
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
		g := util.Gaussian{Mean: mean, Vari: vari}
		algo.Model[fid] = &g
	}
}

func (algo *EPLogisticRegression) Predict(sample *core.Sample) float64 {
	s := util.Gaussian{Mean: 0.0, Vari: 0.0}
	for _, feature := range sample.Features {
		if feature.Value == 0.0 {
			continue
		}
		wi, ok := algo.Model[feature.Id]
		if !ok {
			wi = &(util.Gaussian{Mean: 0.0, Vari: algo.params.init_var})
		}
		s.Mean += feature.Value * wi.Mean
		s.Vari += feature.Value * feature.Value * wi.Vari
	}

	t := s
	t.Vari += algo.params.beta
	return t.Integral(t.Mean / math.Sqrt(t.Vari))
}

func (algo *EPLogisticRegression) Init(params map[string]string) {
	algo.Model = make(map[int64]*util.Gaussian)
	algo.params.beta, _ = strconv.ParseFloat(params["beta"], 64)
	algo.params.init_var = 1.0
}

func (algo *EPLogisticRegression) Clear() {
	algo.Model = nil
	algo.Model = make(map[int64]*util.Gaussian)
}

func (algo *EPLogisticRegression) Train(dataset *core.DataSet) {

	for _, sample := range dataset.Samples {
		s := util.Gaussian{Mean: 0.0, Vari: 0.0}
		for _, feature := range sample.Features {
			if feature.Value == 0.0 {
				continue
			}
			wi, ok := algo.Model[feature.Id]
			if !ok {
				wi = &(util.Gaussian{Mean: 0.0, Vari: algo.params.init_var})
				algo.Model[feature.Id] = wi
			}
			s.Mean += feature.Value * wi.Mean
			s.Vari += feature.Value * feature.Value * wi.Vari
		}

		t := s
		t.Vari += algo.params.beta

		t2 := util.Gaussian{Mean: 0.0, Vari: 0.0}
		if sample.Label > 0.0 {
			t2.UpperTruncateGaussian(t.Mean, t.Vari, 0.0)
		} else {
			t2.LowerTruncateGaussian(t.Mean, t.Vari, 0.0)
		}
		t.MultGaussian(&t2)
		s2 := t
		s2.Vari += algo.params.beta
		s0 := s
		s.MultGaussian(&s2)

		for _, feature := range sample.Features {
			if feature.Value == 0.0 {
				continue
			}
			wi0 := util.Gaussian{Mean: 0.0, Vari: algo.params.init_var}
			w2 := util.Gaussian{Mean: 0.0, Vari: 0.0}
			wi, _ := algo.Model[feature.Id]
			w2.Mean = (s.Mean - (s0.Mean - wi.Mean*feature.Value)) / feature.Value
			w2.Vari = (s.Vari + (s0.Vari - wi.Vari*feature.Value*feature.Value)) / (feature.Value * feature.Value)
			wi.MultGaussian(&w2)
			wi_vari := wi.Vari
			wi_new_vari := wi_vari * wi0.Vari / (0.99*wi0.Vari + 0.01*wi.Vari)
			wi.Vari = wi_new_vari
			wi.Mean = wi.Vari * (0.99*wi.Mean/wi_vari + 0.01*wi0.Mean/wi.Vari)
			if wi.Vari < algo.params.init_var*0.01 {
				wi.Vari = algo.params.init_var * 0.01
			}
			algo.Model[feature.Id] = wi
		}
	}
}
