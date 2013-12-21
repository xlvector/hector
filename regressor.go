package hector

type Regressor interface {

    //Set training parameters from parameter map
    Init(params map[string]string)

    //Train model on a given dataset
    Train(dataset * RealDataSet)

    //Predict the output of an input sample
    Predict(sample * RealSample) float64

    SaveModel(path string)
    LoadModel(path string)
}
