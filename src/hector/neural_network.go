package hector

import(
    "strconv"
    "math/rand"
    "math"
)

type NeuralNetworkParams struct {
    LearningRate float64
    Hidden int64
    Steps int
}

type TwoLayerWeights struct {
    L1 *Matrix
    L2 *Vector
}

type NeuralNetwork struct {
    Model TwoLayerWeights
    Params NeuralNetworkParams
}

func RandomInitVector(dim int64) *Vector{
    v := NewVector()
    var i int64
    for i = 0; i < dim; i++ {
        v.data[i] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
    }
    return v
}

func (algo *NeuralNetwork) Init(params map[string]string) {
    algo.Params.LearningRate, _ = strconv.ParseFloat(params["learning-rate"], 64)
    steps, _ := strconv.ParseInt(params["steps"], 10, 32)
    hidden, _ := strconv.ParseInt(params["hidden"], 10, 64)
    algo.Params.Steps = int(steps)
    algo.Params.Hidden = int64(hidden)
}

func _fn(yi float64)float64{
    return (1-yi)*yi
}

func (algo *NeuralNetwork) Train(dataset * DataSet) {
    algo.Model = TwoLayerWeights{}
    
    algo.Model.L1 = NewMatrix()
    for _, sample := range dataset.Samples {
        for _, f := range sample.Features{
            _, ok := algo.Model.L1.data[f.Id]
            if !ok{
                algo.Model.L1.data[f.Id] = RandomInitVector(algo.Params.Hidden)
            }
        }
    }
    algo.Model.L1 = algo.Model.L1.Trans()

    algo.Model.L2 = NewVector()
    var i int64
    for i = 0; i < algo.Params.Hidden; i++ {
        algo.Model.L2.data[i] = rand.NormFloat64()
    }

    for step := 0; step < algo.Params.Steps; step++{
        for _, sample := range dataset.Samples {
            x := sample.GetFeatureVector()
            y := (algo.Model.L1.MultiplyVector(x)).ApplyOnElem(Sigmoid)
            z := Sigmoid(y.Dot(algo.Model.L2))

            err := sample.LabelDoubleValue() - z
            dL2 := y.Scale(err)
            sig := algo.Model.L2.Scale(err)
            sig = sig.ElemWiseMultiply(y.ApplyOnElem(_fn))
            dL1 := sig.OuterProduct(x)

            algo.Model.L2 = algo.Model.L2.ElemWiseAddVector(dL2.Scale(algo.Params.LearningRate))
            algo.Model.L1 = algo.Model.L1.ElemWiseAddMatrix(dL1.Scale(algo.Params.LearningRate))
        }
    }
}

func (algo *NeuralNetwork) Predict(sample * Sample) float64 {
    return Sigmoid(((algo.Model.L1.MultiplyVector(sample.GetFeatureVector())).ApplyOnElem(Sigmoid)).Dot(algo.Model.L2))
}

