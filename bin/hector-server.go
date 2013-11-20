package main

import (
    "fmt"
    "net/http"
    "encoding/json"
    "hector"
    "strconv"
)

type Context struct {
    feature_iv map[int64]float64
}

type ContextHandler struct {
    c Context
    f ContextFunc
}

type ContextFunc func(c *Context, w http.ResponseWriter, r *http.Request)

func (h ContextHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    h.f(&(h.c), w, r)
}

type Response map[string]interface{}

func (r Response) String() (s string) {
        b, err := json.Marshal(r)
        if err != nil {
            s = ""
            return
        }
        s = string(b)
        return
}

func ToStringMap(m map[int64]float64) map[string]float64 {
	ret := make(map[string]float64)
	for key, value := range m {
		ret[strconv.FormatInt(key, 10)] = value
	}
	return ret
}

func FeatureHandler(c *Context, w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
    fmt.Fprint(w, Response{"success": true, "information_value": ToStringMap(c.feature_iv)})
}

func main() {
	train_path, _, _, _, _ := hector.PrepareParams()
	dataset := hector.NewDataSet()
	dataset.Load(train_path, -1)

	context := Context{feature_iv: hector.InformationValue(dataset)}
	fmt.Println(context)
	handler := ContextHandler{c: context, f: FeatureHandler}
    http.Handle("/", handler)
    http.ListenAndServe(":8080", nil)
}