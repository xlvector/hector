package hector

type Clustering interface {
	Init(params map[string]string)
	Cluster(dataset DataSet)
}