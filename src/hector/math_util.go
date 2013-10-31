package hector

import (
	"math"
)

func Sigmoid(x float64)(y float64) {
	y = 1 / (1 + math.Exp(-1 * x))
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