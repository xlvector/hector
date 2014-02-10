package gp

import (
	"github.com/xlvector/hector/core"
	"math"
)

type CovFunc func(*core.Vector, *core.Vector) float64

func CovMatrix(X []*core.RealSample, cov_func CovFunc) *core.Matrix {
	l := int64(len(X))
	ret := core.NewMatrix()
	for i := int64(0); i < l; i++ {
		for j := i; j < l; j++ {
			c := cov_func(X[i].GetFeatureVector(), X[j].GetFeatureVector())
			ret.SetValue(i, j, c)
			ret.SetValue(j, i, c)
		}
	}
	return ret
}

func CovVector(X []*core.RealSample, y *core.RealSample, cov_func CovFunc) *core.Vector {
	l := int64(len(X))
	ret := core.NewVector()
	for i := int64(0); i < l; i++ {
		ret.SetValue(i, cov_func(X[i].GetFeatureVector(), y.GetFeatureVector()))
	}
	return ret
}

/*
 Squared error covariance function
 ARD = auto relevance detection, and here indicates there is a scaling/radius factor per dimension
*/
type CovSEARD struct {
	Radiuses *core.Vector // dim -> radius
	Amp      float64
}

func (cov_func *CovSEARD) Init(radiuses *core.Vector, amp float64) {
	cov_func.Radiuses = radiuses
	cov_func.Amp = amp
}

func (cov_func *CovSEARD) Cov(x1 *core.Vector, x2 *core.Vector) float64 {
	ret := 0.0
	tmp := 0.0
	for key, r := range cov_func.Radiuses.Data {
		v1 := x1.GetValue(key)
		v2 := x2.GetValue(key)
		tmp = (v1 - v2) / r
		ret += tmp * tmp
	}
	ret = cov_func.Amp * math.Exp(-ret)
	return ret
}
