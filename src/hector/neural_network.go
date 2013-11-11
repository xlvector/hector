package hector

import(
    "strconv"
)

type NeuralNetworkParams struct {
    LearningRate float64
    Hidden int
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

func (algo *NeuralNetwork) Init(params map[string]string) {
    algo.Params.LearningRate, _ = strconv.ParseFloat(params["learning-rate"], 64)
    steps, _ := strconv.ParseInt(params["steps"], 10, 32)
    hidden, _ := strconv.ParseInt(params["hidden"], 10, 32)
    algo.Params.Steps = int(steps)
    algo.Params.Hidden = int(hidden)

    algo.Model = TwoLayerWeights{}
}

func (algo *NeuralNetwork) Predict(sample * Sample) float64 {
    v := NewVector()
    for _, f := range sample.Features {
        v.SetValue(f.Id, f.Value)
    }
    return Sigmoid(((algo.Model.L1.MultiplyVector(v)).ApplyFunc(Sigmoid)).Dot(algo.Model.L2))
}

