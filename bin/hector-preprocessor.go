package main

import(
    "hector"
    "fmt"
    "log"
    "os"
)

func main () {
    train, test, _, _, params := hector.PrepareParams()
    
    action, _ := params["action"]

    if action == "encodelabel" {
        
        fmt.Println("encoded dataset label ..." + train)
        e := hector.NewLabelEncoder()
        EncodeLabelAction(e, train)
        fmt.Println("encoded dataset label ..." + test)
        EncodeLabelAction(e, test)
    }

}

func EncodeLabelAction(e *hector.LabelEncoder, data_path string) {

    dataset := hector.NewDataSet()
    err := dataset.Load(data_path, -1)

    if err != nil{
        log.Fatal(err)
        return
    }
    
    encoded_label_dataset := e.TransformDataset(dataset)
    var output_file *os.File

    output_file, _ = os.Create(data_path + ".hector")
    for _, sample := range encoded_label_dataset.Samples {
        output_file.WriteString(string(sample.ToString(false)) + "\n")
    }

    if output_file != nil{
        defer output_file.Close()
    }
}