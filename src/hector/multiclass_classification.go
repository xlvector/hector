package hector

type MultiClassClassifier struct {
	classifiers []Classifier
	method string
	labels []int
	params map[string]string
}

func (self *MultiClassClassifier) Init(params map[string]string) {
	self.method, _ = params["method"]
	self.params = params
}

func (self *MultiClassClassifier) Train(dataset *DataSet) {
	label_map := make(map[int]bool)
	for _, sample := range dataset.Samples {
		label := int(sample.Label)
		label_map[label] = true
	}

	for label, _ := range label_map {
		self.labels = append(self.labels, label)
		classifer := GetClassifier(self.method)
		classifer.Init(self.params)
		self.classifiers = append(self.classifiers, classifer)
	}

	for i, label := range self.labels {
		subdata := NewDataSet()
			for _, s0 := range dataset.Samples {
				l0 := int(s0.Label)
				s1 := NewSample()
				if l0 == label {
					s1.Label = 1.0
				} else {
					s1.Label = 0.0
				}

				s1.Features = s0.Features
				subdata.AddSample(s1)
			}
			self.classifiers[i].Train(subdata)
	}
}

func (self *MultiClassClassifier) Predict(sample *Sample) map[int]float64 {
	ret := make(map[int]float64)
	for i, classifer := range self.classifiers {
		p := classifer.Predict(sample)
		ret[self.labels[i]] = p
	}
	return ret
}

func RunMultiClassClassification(train_dataset *DataSet, test_dataset *DataSet, params map[string]string) float64{
	classifer := MultiClassClassifier{}
	classifer.Init(params)

	classifer.Train(train_dataset)

	match := 0.0
	total := 0.0
	for _, sample := range test_dataset.Samples {
		pred := classifer.Predict(sample)
		label := int(sample.Label)

		max_weight := 0.0
		max_label := 0

		for label, w := range pred {
			if max_weight < w {
				max_weight = w
				max_label = label
			}
		}
		if max_label == label{
			match += 1.0
		}
		total += 1.0
	}
	return match / total
}