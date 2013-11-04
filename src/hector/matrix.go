package hector

type Matrix struct {
	data map[int64]*Vector
}

func NewMatrix() *Matrix {
	m := Matrix{}
	m.data = make(map[int64]*Vector)
	return &m
}

func (m *Matrix) AddValue(k1, k2 int64, v float64){
	_, ok := m.data[k1]
	if !ok {
		m.data[k1] = NewVector()
	}
	m.data[k1].AddValue(k2, v)
}

func (m *Matrix) SetValue(k1, k2 int64, v float64){
	_, ok := m.data[k1]
	if !ok {
		m.data[k1] = NewVector()
	}
	m.data[k1].SetValue(k2, v)
}

func (m *Matrix) GetRow(k1 int64) *Vector {
	row, ok := m.data[k1]
	if !ok {
		return nil
	} else {
		return row
	}
	return row
}
