package util

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
	Mean, Vari float64
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
	g.Mean += g1.Mean
	g.Vari += g1.Vari
}

func (g *Gaussian) MultGaussian(g1 *Gaussian){
	Mean := (g.Mean * g1.Vari + g1.Mean * g.Vari) / (g.Vari + g1.Vari)
	Vari := g.Vari * g1.Vari / (g.Vari + g1.Vari)
	g.Mean = Mean
	g.Vari = Vari
}

func (g *Gaussian) Func(x float64) float64{
	return math.Exp(-0.5 * x * x) * 0.3989423;
}

func (g *Gaussian) UpperTruncateGaussian(Mean, Vari, s float64){
	sqrtVari := math.Sqrt(Vari)
	a := (s - Mean) / sqrtVari
	lambda := a
	if a < 4.0 {
		lambda = g.Func(a) / g.Integral(-1.0 * a)
	}
	Mean = Mean + sqrtVari * lambda
	if lambda * (lambda - a) > 1{
		Vari = 0.0
	} else {
		Vari *= 1 - lambda * (lambda - a)
	}
	g.Mean = Mean
	g.Vari = Vari
}

func (g *Gaussian) LowerTruncateGaussian(Mean, Vari, s float64){
	sqrtVari := math.Sqrt(Vari)
	a := (s - Mean) / sqrtVari
	delta := -1.0 * a
	if a > -4.0 {
		delta = g.Func(a) / g.Integral(a)
	}
	Mean = Mean - sqrtVari * delta
	if a * delta + delta * delta > 1.0 {
		Vari = 0.0
	} else {
		Vari *= 1 - a * delta - delta * delta
	}
	g.Mean = Mean
	g.Vari = Vari
}