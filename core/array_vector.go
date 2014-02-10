package core

import (
	"github.com/xlvector/hector/util"
	"math"
	"strconv"
	"strings"
)

type ArrayVector struct {
	data []float64
}

func NewArrayVector() *ArrayVector {
	v := ArrayVector{}
	v.data = []float64{}
	return &v
}

func (v *ArrayVector) ToString() []byte {
	sb := util.StringBuilder{}
	for _, value := range v.data {
		sb.Float(value)
		sb.Write("|")
	}
	return sb.Bytes()
}

func (v *ArrayVector) FromString(buf string) {
	tks := strings.Split(buf, "|")
	for _, tk := range tks {
		if len(tk) == 0 {
			continue
		}
		value, _ := strconv.ParseFloat(tk, 64)
		v.data = append(v.data, value)
	}
}

func (v *ArrayVector) Expand(size int) {
	for len(v.data) < size {
		v.data = append(v.data, 0.0)
	}
}

func (v *ArrayVector) AddValue(key int, value float64) {
	v.Expand(key + 1)
	v.data[key] += value
}

func (v *ArrayVector) GetValue(key int) float64 {
	if key >= len(v.data) {
		return 0.0
	} else {
		return v.data[key]
	}
}

func (v *ArrayVector) SetValue(key int, value float64) {
	v.Expand(key + 1)
	v.data[key] = value
}

func (v *ArrayVector) AddVector(v2 *ArrayVector, alpha float64) {
	for key, value := range v2.data {
		v.AddValue(key, value*alpha)
	}
}

func (v *ArrayVector) NormL2() float64 {
	ret := 0.0
	for _, val := range v.data {
		ret += val * val
	}
	return ret
}

func (v *ArrayVector) Copy() *ArrayVector {
	ret := NewArrayVector()
	for key, val := range v.data {
		ret.SetValue(key, val)
	}
	return ret
}

func (v *ArrayVector) KeyWithMaxValue() (int, float64) {
	ret := 0
	max_val := 0.0
	for key, val := range v.data {
		max_val = val
		ret = key
		break
	}
	for key, val := range v.data {
		if max_val < val {
			max_val = val
			ret = key
		}
	}
	return ret, max_val
}

func (v *ArrayVector) Sum() float64 {
	ret := 0.0
	for _, val := range v.data {
		ret += val
	}
	return ret
}

func (v *ArrayVector) Dot(v2 *ArrayVector) float64 {
	va := v
	vb := v2

	if len(v2.data) < len(v.data) {
		va = v2
		vb = v
	}
	ret := 0.0
	for key, a := range va.data {
		b := vb.data[key]
		ret += a * b
	}
	return ret
}

func (v *ArrayVector) Scale(s float64) {
	for i, _ := range v.data {
		v.data[i] *= s
	}
}

func (v *ArrayVector) SoftMaxNorm() *ArrayVector {
	sum := 0.0
	for _, val := range v.data {
		sum += math.Exp(val)
	}
	ret := NewArrayVector()
	for key, val := range v.data {
		ret.SetValue(key, math.Exp(val)/sum)
	}
	return ret
}
