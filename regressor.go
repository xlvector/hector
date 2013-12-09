package hector

type Regressor interface {

    //Set training parameters from parameter map
    Init(params map[string]string)

    //Train model on a given dataset
    Train(dataset * DataSet)

    //Predict the output of an input sample
    Predict(sample * Sample) float64

    SaveModel(path string)
    LoadModel(path string)
}
