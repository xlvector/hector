package hector

import(
    "math"
    "strings"
    "os"
    "bufio"
    "strconv"
)

type LROWLQNParams struct {
    Regularization float64
}

type LROWLQN struct {
    Model *Vector
    Params LROWLQNParams
    // for training
    dataSet *DataSet
    lastPos *Vector
    lastCost float64
    lastGrad *Vector
}

func (lr *LROWLQN) SaveModel(path string) {
    sb := StringBuilder{}
    for key, val := range lr.Model.data {
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
    lr.Model = NewVector()
    lr.Params.Regularization, _ = strconv.ParseFloat(params["regularization"], 64)
}

func (lr *LROWLQN) updateValueGrad(pos *Vector, dataset *DataSet) {
    var totalLoss float64 = 0.0
    var grad *Vector = NewVector()
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
            grad.AddValue(fea.Id, scale * fea.Value)
        }
    }
    lr.lastPos = pos.Copy()
    lr.lastCost = totalLoss
    lr.lastGrad = grad
}

func (lr *LROWLQN) Equals(x *Vector, y *Vector) bool {
    if y == nil && x == nil {
        return true
    }
    if y == nil || x == nil {
        return false
    }
    for key, val := range x.data {
        if y.GetValue(key) != val {
            return false
        }
    }
    for key, val := range y.data {
        if x.GetValue(key) != val {
            return false
        }
    }
    return true
}

func (lr *LROWLQN) Value(pos *Vector) float64 {
    if lr.Equals(pos, lr.lastPos) {
        return lr.lastCost
    }
    lr.updateValueGrad(pos, lr.dataSet)
    return lr.lastCost
}

func (lr *LROWLQN) Gradient(pos *Vector) *Vector{
    if lr.Equals(pos, lr.lastPos) {
        return lr.lastGrad
    }
    lr.updateValueGrad(pos, lr.dataSet)
    return lr.lastGrad
}

func (lr *LROWLQN) Train(dataset *DataSet) {
    lr.dataSet = dataset
    minimizer := NewOWLQNMinimizer(lr.Params.Regularization)
    lr.Model = minimizer.Minimize(lr, NewVector())
}

func (lr *LROWLQN) getScore(model *Vector, sample *Sample) float64 {
    var score float64 = 0
    for _, fea := range sample.Features {
        score += model.GetValue(fea.Id) * fea.Value
    }
    return score
}

func (lr *LROWLQN) Predict(sample *Sample) float64 {
    score := lr.getScore(lr.Model, sample)
    score = 1.0 / (1.0 + math.Exp(-score))
    return score
}
