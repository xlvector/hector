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

type LROWLQNParams struct {
	Regularization float64
}

type LROWLQN struct {
	Model  *core.Vector
	Params LROWLQNParams
	// for training
	dataSet  *core.DataSet
	lastPos  *core.Vector
	lastCost float64
	lastGrad *core.Vector
}

func (lr *LROWLQN) SaveModel(path string) {
	sb := util.StringBuilder{}
	for key, val := range lr.Model.Data {
		sb.Int64(key)
		sb.Write("\t")
		sb.Float(val)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (lr *LROWLQN) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		key, _ := strconv.ParseInt(tks[0], 10, 64)
		val, _ := strconv.ParseFloat(tks[1], 64)
		lr.Model.SetValue(key, val)
	}
}

func (lr *LROWLQN) Init(params map[string]string) {
	lr.Model = core.NewVector()
	lr.Params.Regularization, _ = strconv.ParseFloat(params["regularization"], 64)
}

func (lr *LROWLQN) updateValueGrad(pos *core.Vector, dataset *core.DataSet) {
	var totalLoss float64 = 0.0
	var grad *core.Vector = core.NewVector()
	for _, sample := range dataset.Samples {
		var score float64 = lr.getScore(pos, sample)
		var signScore float64 = score
		if sample.Label == 0 {
			signScore = -score
		}
		var prob float64
		var lnProb float64
		if signScore < -30 {
			prob = 0
			lnProb = signScore
		} else if signScore > 30 {
			prob = 1
			lnProb = 0
		} else {
			prob = 1.0 / (1.0 + math.Exp(-signScore))
			lnProb = math.Log(prob)
		}
		var scale float64
		if sample.Label == 0 {
			scale = (1 - prob)
		} else {
			scale = -(1 - prob)
		}
		totalLoss += -lnProb
		for _, fea := range sample.Features {
			grad.AddValue(fea.Id, scale*fea.Value)
		}
	}
	lr.lastPos = pos.Copy()
	lr.lastCost = totalLoss
	lr.lastGrad = grad
}

func (lr *LROWLQN) Equals(x *core.Vector, y *core.Vector) bool {
	if y == nil && x == nil {
		return true
	}
	if y == nil || x == nil {
		return false
	}
	for key, val := range x.Data {
		if y.GetValue(key) != val {
			return false
		}
	}
	for key, val := range y.Data {
		if x.GetValue(key) != val {
			return false
		}
	}
	return true
}

func (lr *LROWLQN) Value(pos *core.Vector) float64 {
	if lr.Equals(pos, lr.lastPos) {
		return lr.lastCost
	}
	lr.updateValueGrad(pos, lr.dataSet)
	return lr.lastCost
}

func (lr *LROWLQN) Gradient(pos *core.Vector) *core.Vector {
	if lr.Equals(pos, lr.lastPos) {
		return lr.lastGrad
	}
	lr.updateValueGrad(pos, lr.dataSet)
	return lr.lastGrad
}

func (lr *LROWLQN) Train(dataset *core.DataSet) {
	lr.dataSet = dataset
	minimizer := NewOWLQNMinimizer(lr.Params.Regularization)
	lr.Model = minimizer.Minimize(lr, core.NewVector())
}

func (lr *LROWLQN) getScore(model *core.Vector, sample *core.Sample) float64 {
	var score float64 = 0
	for _, fea := range sample.Features {
		score += model.GetValue(fea.Id) * fea.Value
	}
	return score
}

func (lr *LROWLQN) Predict(sample *core.Sample) float64 {
	score := lr.getScore(lr.Model, sample)
	score = 1.0 / (1.0 + math.Exp(-score))
	return score
}
