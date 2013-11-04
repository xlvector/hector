package hector

import (
	"math"
	"strconv"
)

func Sigmoid(x float64)(y float64) {
	y = 1 / (1 + math.Exp(-1 * x))
	return y
}

func UnSigmoid(x float64) float64 {
	x = x * 0.99 + 0.01
	y := math.Log(x / (1 - x))
	return y
}

func Signum(x float64) float64 {
	ret := 0.0
	if x > 0{
		ret = 1.0
	} else if(x < 0) {
		ret = -1.0
	} else {
		ret = 0.0
	}
	return ret
}

func ParseInt64(str string) int64 {
	ret, _ := strconv.ParseInt(str, 10, 64)
	return ret
}

func ParseFloat64(str string) float64 {
	ret, _ := strconv.ParseFloat(str, 64)
	return ret
}

type Gaussian struct {
	mean, vari float64
}

func (g *Gaussian) Integral(x float64) float64{
	a1 := 0.254829592
	a2 := -0.284496736
	a3 := 1.421413741
	a4 := -1.453152027
	a5 := 1.061405429
	p := 0.3275911

	sign := 1.0
	if x < 0{
		sign = -1.0
	}
	x = math.Abs(x) / math.Sqrt(2.0)

	t := 1.0 / (1.0 + p * x)
	y := 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.Exp(-x * x)
	return 0.5 * (1.0 + sign * y)
}

func (g *Gaussian) AddGaussian(g1 *Gaussian){
	g.mean += g1.mean
	g.vari += g1.vari
}

func (g *Gaussian) MultGaussian(g1 *Gaussian){
	mean := (g.mean * g1.vari + g1.mean * g.vari) / (g.vari + g1.vari)
	vari := g.vari * g1.vari / (g.vari + g1.vari)
	g.mean = mean
	g.vari = vari
}

func (g *Gaussian) Func(x float64) float64{
	return math.Exp(-0.5 * x * x) * 0.3989423;
}

func (g *Gaussian) UpperTruncateGaussian(mean, vari, s float64){
	sqrtvari := math.Sqrt(vari)
	a := (s - mean) / sqrtvari
	lambda := a
	if a < 4.0 {
		lambda = g.Func(a) / g.Integral(-1.0 * a)
	}
	mean = mean + sqrtvari * lambda
	if lambda * (lambda - a) > 1{
		vari = 0.0
	} else {
		vari *= 1 - lambda * (lambda - a)
	}
	g.mean = mean
	g.vari = vari
}

func (g *Gaussian) LowerTruncateGaussian(mean, vari, s float64){
	sqrtvari := math.Sqrt(vari)
	a := (s - mean) / sqrtvari
	delta := -1.0 * a
	if a > -4.0 {
		delta = g.Func(a) / g.Integral(a)
	}
	mean = mean - sqrtvari * delta
	if a * delta + delta * delta > 1.0 {
		vari = 0.0
	} else {
		vari *= 1 - a * delta - delta * delta
	}
	g.mean = mean
	g.vari = vari
}