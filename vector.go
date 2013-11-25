package hector

import (
	"math/rand"
	"strings"
	"strconv"
)

type Vector struct {
	data map[int64]float64
}

func NewVector() *Vector {
	v := Vector{}
	v.data = make(map[int64]float64)
	return &v
}

func (v *Vector) ToString() []byte {
	sb := StringBuilder{}
	for key, value := range v.data {
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
		key,_ := strconv.ParseInt(kv[0], 10, 64)
		value,_ := strconv.ParseFloat(kv[1], 64)
		v.data[key] = value
	}
}

func (v *Vector) AddValue(key int64, value float64) {
	_, ok := v.data[key]
	if ok{
		v.data[key] += value
	} else {
		v.data[key] = value
	}
}

func (v *Vector) GetValue(key int64) float64{
	value, ok := v.data[key]
	if !ok {
		return 0.0
	} else {
		return value
	}
}

func (v *Vector) RandomInit(key int64, c float64){
	value, ok := v.data[key]
	if !ok {
		value = rand.NormFloat64() * c
		v.data[key] = value
	}
}

func (v *Vector) SetValue(key int64, value float64) {
	v.data[key] = value
}

func (v *Vector) AddVector(v2 *Vector, alpha float64) {
	for key, value := range v2.data {
		v.AddValue(key, value * alpha)
	}
}

func (v *Vector) NormL2() float64{
	ret := 0.0
	for _, val := range v.data{
		ret += val * val
	}
	return ret
}

func (v *Vector) Copy() *Vector{
	ret := NewVector()
	for key, val := range v.data {
		ret.SetValue(key, val)
	}
	return ret
}

func (v *Vector) KeyWithMaxValue() (int64, float64) {
	ret := int64(0)
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

func (v *Vector) Sum() float64 {
	ret := 0.0
	for _, val := range v.data {
		ret += val
	}
	return ret
}

func (v *Vector) Dot(v2 *Vector) float64{
	va := v
	vb := v2

	if len(v2.data) < len(v.data) {
		va = v2
		vb = v
	}
	ret := 0.0
	for key, a := range va.data{
		b, ok := vb.data[key]
		if ok {
			ret += a*b
		}
	}
	return ret	
}

func (v *Vector) DotFeatures(fs []Feature) float64{
	ret := 0.0
	for _, f := range fs{
		ret += f.Value * v.GetValue(f.Id)
	}
	return ret
}

type ElemOperation func(float64)(float64)

func (v *Vector) ApplyOnElem(fn ElemOperation) *Vector{
	ret := NewVector()
	for key, val := range v.data{
		ret.SetValue(key, fn(val))
	}
	return ret
}

func (v *Vector) Scale(scale float64) *Vector{
	ret := NewVector()
	for key, val := range v.data{
		ret.SetValue(key, val * scale)
	}
	return ret
}

func (v *Vector) ApplyScale(scale float64) {
	for key, val := range v.data {
		v.data[key] = val * scale
	}
}


func (v *Vector) SoftMaxNorm() *Vector {
	sum := 0.0
	for key, val := range v.data {
		sum += math.Exp(val)
	}
	ret := NewVector()
	for key, val := range v.data {
		ret.SetValue(key, math.Exp(val) / sum)
	}
	return ret
}

func (v *Vector) ElemWiseAddVector(u *Vector) *Vector{
	ret := NewVector()
	for key, vi := range v.data{
		ret.SetValue(key, vi)
	}
	for key, ui := range u.data{
		ret.AddValue(key, ui)
	}
	return ret
}

func (v *Vector) ElemWiseMultiply(u *Vector) *Vector{
	ret := NewVector()
	for key, val := range v.data{
		ual := u.GetValue(key)
		if ual != 0 && val !=0{
			ret.SetValue(key, val*ual)
		}
	}
	return ret
}

func (v *Vector) ElemWiseMultiplyAdd(u *Vector, s float64) *Vector {
	ret := NewVector()
	for key, val := range v.data {
		ret.SetValue(key, val)
	}
	for key, val := range u.data {
		ret.AddValue(key, val * s)
	}
	return ret
}

func (v *Vector) ApplyElemWiseMultiplyAccumulation(u *Vector, s float64) {
	for key, val := range u.data {
		v.AddValue(val * s)
	}
}

func (v *Vector) OuterProduct(u *Vector) *Matrix{
	ret := NewMatrix()
	for key, vi := range v.data{
		ret.data[key] = u.Scale(vi)
	}
	return ret
}
