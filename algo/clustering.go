package algo

import (
	"github.com/hector/core"
)

type Clustering interface {
	Init(params map[string]string)
	Cluster(dataset core.DataSet)
}
