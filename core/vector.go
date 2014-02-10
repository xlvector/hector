package core

import (
	"github.com/xlvector/hector/util"
	"math"
	"math/rand"
	"strconv"
	"strings"
)

type Vector struct {
	Data map[int64]float64
}

func NewVector() *Vector {
	v := Vector{}
	v.Data = make(map[int64]float64)
	return &v
}

func (v *Vector) ToString() []byte {
	sb := util.StringBuilder{}
	for key, value := range v.Data {
		sb.Int64(key)
		sb.Write(":")
		sb.Float(value)
		sb.Write("|")
	}
	return sb.Bytes()
}

func (v *Vector) FromString(buf string) {
	tks := strings.Split(buf, "|")
	for _, tk := range tks {
		if len(tk) == 0 {
			continue
		}
		kv := strings.Split(tk, ":")
		key, _ := strconv.ParseInt(kv[0], 10, 64)
		value, _ := strconv.ParseFloat(kv[1], 64)
		v.Data[key] = value
	}
}

func (v *Vector) AddValue(key int64, value float64) {
	_, ok := v.Data[key]
	if ok {
		v.Data[key] += value
	} else {
		v.Data[key] = value
	}
}

func (v *Vector) GetValue(key int64) float64 {
	value, ok := v.Data[key]
	if !ok {
		return 0.0
	} else {
		return value
	}
}

func (v *Vector) RandomInit(key int64, c float64) {
	value, ok := v.Data[key]
	if !ok {
		value = rand.NormFloat64() * c
		v.Data[key] = value
	}
}

func (v *Vector) SetValue(key int64, value float64) {
	v.Data[key] = value
}

func (v *Vector) AddVector(v2 *Vector, alpha float64) {
	for key, value := range v2.Data {
		v.AddValue(key, value*alpha)
	}
}

func (v *Vector) NormL2() float64 {
	ret := 0.0
	for _, val := range v.Data {
		ret += val * val
	}
	return ret
}

func (v *Vector) Copy() *Vector {
	ret := NewVector()
	for key, val := range v.Data {
		ret.SetValue(key, val)
	}
	return ret
}

func (v *Vector) KeyWithMaxValue() (int64, float64) {
	ret := int64(0)
	max_val := 0.0
	for key, val := range v.Data {
		max_val = val
		ret = key
		break
	}
	for key, val := range v.Data {
		if max_val < val {
			max_val = val
			ret = key
		}
	}
	return ret, max_val
}

func (v *Vector) Sum() float64 {
	ret := 0.0
	for _, val := range v.Data {
		ret += val
	}
	return ret
}

func (v *Vector) Dot(v2 *Vector) float64 {
	va := v
	vb := v2

	if len(v2.Data) < len(v.Data) {
		va = v2
		vb = v
	}
	ret := 0.0
	for key, a := range va.Data {
		b, ok := vb.Data[key]
		if ok {
			ret += a * b
		}
	}
	return ret
}

func (v *Vector) DotFeatures(fs []Feature) float64 {
	ret := 0.0
	for _, f := range fs {
		ret += f.Value * v.GetValue(f.Id)
	}
	return ret
}

type ElemOperation func(float64) float64

func (v *Vector) ApplyOnElem(fn ElemOperation) *Vector {
	ret := NewVector()
	for key, val := range v.Data {
		ret.SetValue(key, fn(val))
	}
	return ret
}

func (v *Vector) Scale(scale float64) *Vector {
	ret := NewVector()
	for key, val := range v.Data {
		ret.SetValue(key, val*scale)
	}
	return ret
}

func (v *Vector) ApplyScale(scale float64) {
	for key, val := range v.Data {
		v.Data[key] = val * scale
	}
}

func (v *Vector) SoftMaxNorm() *Vector {
	sum := 0.0
	for _, val := range v.Data {
		sum += math.Exp(val)
	}
	ret := NewVector()
	for key, val := range v.Data {
		ret.SetValue(key, math.Exp(val)/sum)
	}
	return ret
}

func (v *Vector) ElemWiseAddVector(u *Vector) *Vector {
	ret := NewVector()
	for key, vi := range v.Data {
		ret.SetValue(key, vi)
	}
	for key, ui := range u.Data {
		ret.AddValue(key, ui)
	}
	return ret
}

func (v *Vector) ElemWiseMultiply(u *Vector) *Vector {
	ret := NewVector()
	for key, val := range v.Data {
		ual := u.GetValue(key)
		if ual != 0 && val != 0 {
			ret.SetValue(key, val*ual)
		}
	}
	return ret
}

func (v *Vector) ElemWiseMultiplyAdd(u *Vector, s float64) *Vector {
	ret := NewVector()
	for key, val := range v.Data {
		ret.SetValue(key, val)
	}
	for key, val := range u.Data {
		ret.AddValue(key, val*s)
	}
	return ret
}

func (v *Vector) ApplyElemWiseMultiplyAccumulation(u *Vector, s float64) {
	for key, val := range u.Data {
		v.AddValue(key, val*s)
	}
}

func (v *Vector) OuterProduct(u *Vector) *Matrix {
	ret := NewMatrix()
	for key, vi := range v.Data {
		ret.Data[key] = u.Scale(vi)
	}
	return ret
}

func (v *Vector) MultiplyMatrix(m *Matrix) *Vector {
	ret := NewVector()
	for k, v := range v.Data {
		u, ok := m.Data[k]
		if ok {
			for ki, ui := range u.Data {
				ret.Data[ki] += v * ui
			}
		}
	}
	return ret
}
