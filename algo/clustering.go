package algo

import (
    "hector/core"
)

type Clustering interface {
    Init(params map[string]string)
    Cluster(dataset core.DataSet)
}