package main

import (
	"encoding/json"
	"fmt"
	"github.com/xlvector/hector"
	"github.com/xlvector/hector/algo"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
	"log"
	"net/http"
)

type ClassifierHandler struct {
	classifier algo.Classifier
}

func (c *ClassifierHandler) ServeHTTP(w http.ResponseWriter,
	req *http.Request) {
	sample := core.NewSample()
	if req.Method != "POST" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	features := req.FormValue("features")
	if len(features) == 0 {
		http.Error(w, "need input features", http.StatusInternalServerError)
		return
	}
	fs := make(map[string]float64)
	err := json.Unmarshal([]byte(features), &fs)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for k, v := range fs {
		f := core.Feature{
			Id:    util.Hash(k),
			Value: v,
		}
		sample.AddFeature(f)
	}
	p := c.classifier.Predict(sample)
	output, err := json.Marshal(map[string]interface{}{
		"prediction": p,
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	fmt.Fprint(w, output)
}

func main() {
	_, _, _, method, params := hector.PrepareParams()
	ch := &ClassifierHandler{
		classifier: hector.GetClassifier(method),
	}
	model, ok := params["model"]
	if !ok {
		log.Fatalln("please input model file")
	}
	ch.classifier.LoadModel(model)
	http.Handle("/predict", ch)
	err := http.ListenAndServe(":"+params["port"], nil)
	if err != nil {
		log.Fatal(err)
	}
}
