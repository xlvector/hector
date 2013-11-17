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

func (self *NeuralNetwork) SaveModel(path string){

}

func (self *NeuralNetwork) LoadModel(path string){
    
}

func (algo *NeuralNetwork) Init(params map[string]string) {
    algo.Params.LearningRate, _ = strconv.ParseFloat(params["learning-rate"], 64)
    steps, _ := strconv.ParseInt(params["steps"], 10, 32)
    hidden, _ := strconv.ParseInt(params["hidden"], 10, 64)
    algo.Params.Steps = int(steps)
    algo.Params.Hidden = int64(hidden)
}

func (algo *NeuralNetwork) Train(dataset * DataSet) {
    algo.Model = TwoLayerWeights{}

    algo.Model.L1 = NewMatrix()
    for i := int64(0); i < algo.Params.Hidden; i++ {
        algo.Model.L1.data[i] = NewVector()
    }
    
    initalized := make(map[int64]int)
    for _, sample := range dataset.Samples {
        for _, f := range sample.Features{
            _, ok := initalized[f.Id]
            if !ok{
                for i := int64(0); i < algo.Params.Hidden; i++ {
                    algo.Model.L1.SetValue(i, f.Id, (rand.Float64() - 0.5) / math.Sqrt(float64(algo.Params.Hidden)))               
                }
                initalized[f.Id] = 1
            }
        }
    }
    
    algo.Model.L2 = NewVector()
    for i := int64(0); i < algo.Params.Hidden; i++ {
        algo.Model.L2.data[i] = rand.NormFloat64()
    }

    for step := 0; step < algo.Params.Steps; step++{
        for _, sample := range dataset.Samples {
            y := NewVector()
            z := float64(0)
            for i := int64(0); i < algo.Params.Hidden; i++ {
                sum := float64(0)
                for _, f := range sample.Features {
                    sum += f.Value * algo.Model.L1.data[i].GetValue(f.Id)
                }
                y.data[i] = Sigmoid(sum)
                z += (y.data[i] * algo.Model.L2.data[i])
            }
            z = Sigmoid(z)

            err := sample.LabelDoubleValue() - z
            sig := NewVector()
            for key, val := range y.data {
                sig.SetValue(key, err * algo.Model.L2.GetValue(key) * (1-val) * val)
            }
            for key, val := range algo.Model.L2.data {
                algo.Model.L2.SetValue(key, val + algo.Params.LearningRate * y.GetValue(key) * err)
            }
            for i, s := range sig.data {
                if s != 0 {
                    for _, f := range sample.Features {
                        val := algo.Model.L1.data[i].GetValue(f.Id)
                        algo.Model.L1.SetValue(i, f.Id, val + algo.Params.LearningRate * s * f.Value)
                    }
                }
            }
        }
    }
}

func (algo *NeuralNetwork) Predict(sample * Sample) float64 {
    return Sigmoid(((algo.Model.L1.MultiplyVector(sample.GetFeatureVector())).ApplyOnElem(Sigmoid)).Dot(algo.Model.L2))
}

