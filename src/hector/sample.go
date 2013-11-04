package hector

type RawSample struct {
	Features map[string]string
	Label float64
}

func NewRawSample() *RawSample {
	ret := RawSample{}
	ret.Features = make(map[string]string)
	ret.Label = 0.0
	return &ret
}

func (s *RawSample) GetFeatureValue(key string) string {
	value, ok := s.Features[key]
	if ok {
		return value
	} else {
		return "nil"
	}
}

type Sample struct {
	Features []Feature
	Label float64
}

func NewSample() *Sample {
	ret := Sample{}
	ret.Features = []Feature{}
	ret.Label = 0.0
	return &ret
}

func (s *Sample) LabelDoubleValue() float64 {
	return float64(s.Label)
}

func (s *Sample) AddFeature(f Feature){
	s.Features = append(s.Features, f)
}

type MapBasedSample struct {
	Features map[int64]float64
	Label float64	
}

func (s *MapBasedSample) LabelDoubleValue() float64 {
	return float64(s.Label)
}

func (s *Sample) ToMapBasedSample() *MapBasedSample {
	ret := MapBasedSample{}
	ret.Features = make(map[int64]float64)
	ret.Label = s.Label
	for _, feature := range s.Features{
		ret.Features[feature.Id] = feature.Value
	}
	return &ret	
}

func (s *Sample) GetFeatureVector() *Vector {
	ret := NewVector()
	for _, f := range s.Features {
		ret.SetValue(f.Id, f.Value)
	}
	return ret
}

