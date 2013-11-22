package hector

import(
    "strconv"
    "math/rand"
    "math"
    "fmt"
)

type NeuralNetworkParams struct {
    LearningRate float64
    Regularization float64
    Hidden int64
    Steps int
}

type TwoLayerWeights struct {
    L1 *Matrix
    L2 *Matrix
}


/*
Please refer to this chapter to know algorithm details : 
http://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
*/
type NeuralNetwork struct {
    Model TwoLayerWeights
    MaxLabel int
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
    algo.Params.Regularization, _ = strconv.ParseFloat(params["regularization"], 64)
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
    max_label := 0
    for _, sample := range dataset.Samples {
        if max_label < sample.Label{
            max_label = sample.Label
        }
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
    algo.MaxLabel = max_label
    
    algo.Model.L2 = NewMatrix()
    for i := int64(0); i < algo.Params.Hidden; i++ {
        for j := int64(0); j <= int64(max_label); j++ {
            algo.Model.L2.SetValue(i, j, (rand.NormFloat64() / math.Sqrt(float64(max_label) + 1.0)))
        }
    }

    for step := 0; step < algo.Params.Steps; step++{
        fmt.Printf(".")
        for _, sample := range dataset.Samples {
            y := NewVector()
            z := NewVector()
            for i := int64(0); i < algo.Params.Hidden; i++ {
                sum := float64(0)
                for _, f := range sample.Features {
                    sum += f.Value * algo.Model.L1.data[i].GetValue(f.Id)
                }
                y.data[i] = Sigmoid(sum)
                for j := int64(0); j <= int64(max_label); j++ {
                    z.AddValue(j, y.GetValue(i) * algo.Model.L2.GetValue(i, j))
                }
            }
            z = z.SoftMaxNorm()

            err := NewVector()
            err.AddValue(int64(sample.Label), 1.0)
            err.AddVector(z, -1.0)

            delta_hidden := NewVector()
            for i := int64(0); i < algo.Params.Hidden; i++ {
                out_i := y.GetValue(i)
                for j := int64(0); j <= int64(max_label); j++ {
                    wij := algo.Model.L2.GetValue(i, j)
                    delta_j := err.GetValue(j)
                    
                    wij += algo.Params.LearningRate * (out_i * delta_j - algo.Params.Regularization * wij)
                    algo.Model.L2.SetValue(i, j, wij)
                    delta_hidden.AddValue(i, out_i * (1.0 - out_i) * delta_j * wij)
                }
            }

            for _, f := range sample.Features {
                for j := int64(0); j < algo.Params.Hidden; j++ {
                    wij := algo.Model.L1.GetValue(j, f.Id)
                    wij += algo.Params.LearningRate * (delta_hidden.GetValue(j) * f.Value - algo.Params.Regularization * wij)
                    algo.Model.L1.SetValue(j, f.Id, wij)
                }
            }
        }
    }
    fmt.Println()
}

func (algo *NeuralNetwork) PredictMultiClass(sample * Sample) * ArrayVector {
    y := NewVector()
    z := NewArrayVector()
    for i := int64(0); i < algo.Params.Hidden; i++ {
        sum := float64(0)
        for _, f := range sample.Features {
            sum += f.Value * algo.Model.L1.data[i].GetValue(f.Id)
        }
        y.data[i] = Sigmoid(sum)
        for j := 0; j <= algo.MaxLabel; j++ {
            z.AddValue(j, y.GetValue(i) * algo.Model.L2.GetValue(i, int64(j)))
        }
    }
    z = z.SoftMaxNorm()
    return z
}

func (algo *NeuralNetwork) Predict(sample *Sample) float64 {
    z := algo.PredictMultiClass(sample)
    return z.GetValue(1)
}

