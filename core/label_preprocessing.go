package core

import(
    "fmt"
)

type IntEncoder struct {
    Mapping map[int]int
    InverseMapping map[int]int
}

func NewIntEncoder() *IntEncoder {
    e := IntEncoder{}
    e.Mapping = make(map[int]int)
    e.InverseMapping = make(map[int]int)
    return &e
}

func (e *IntEncoder) Encoded(original int) int{
    if encoded, ok := e.Mapping[original]; ok {
        return encoded    
    }

    e.Mapping[original] = len(e.Mapping)
    encoded := e.Mapping[original]
    e.InverseMapping[encoded] = original            
    return encoded
}

func (e *IntEncoder) Decoded(encoded int) (int, error){
    if decoded, ok := e.InverseMapping[encoded]; ok {
        return decoded, nil  
    }

    return -1, fmt.Errorf("Can't find %d in dictionary...", encoded)
}

type LabelEncoder struct { 
    labelMapper *IntEncoder
}

func NewLabelEncoder() *LabelEncoder {
    e := LabelEncoder{}
    e.labelMapper = NewIntEncoder()
    return &e
}

func (e *LabelEncoder) TransformSample(s *Sample) *Sample{
    ret := s.Clone()
    ret.Label = e.labelMapper.Encoded(ret.Label)
    return ret
}

func (e *LabelEncoder) TransformDataset(dataset *DataSet) *DataSet{
    ret := NewDataSet()
    for _, sample := range dataset.Samples {
        ret.AddSample(e.TransformSample(sample))
    }

    return ret
}

func (e *LabelEncoder) InverseTransformSample(s *Sample) *Sample{
    ret := s.Clone()
    ret.Label, _ = e.labelMapper.Decoded(ret.Label)
    return ret
}

func (e *LabelEncoder) InverseTransformDataset(dataset *DataSet) *DataSet{
    ret := NewDataSet()
    for _, sample := range dataset.Samples {
        ret.AddSample(e.InverseTransformSample(sample))
    }

    return ret
}