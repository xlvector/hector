package svm

import (
	"fmt"
	"github.com/xlvector/hector/core"
	"math"
	"math/rand"
	"strconv"
)

type SVM struct {
	sv []*core.Vector
	y  []float64
	a  []float64
	b  float64
	C  float64
	e  float64
	w  *core.Vector

	xx []float64
}

func (self *SVM) SaveModel(path string) {

}

func (self *SVM) LoadModel(path string) {

}

type SVMValues struct {
	a1, a2, e1, e2, k11, k12, k22 float64
	i1, i2                        int
}

func (c *SVM) Init(params map[string]string) {
	c.C, _ = strconv.ParseFloat(params["c"], 64)
	c.e, _ = strconv.ParseFloat(params["e"], 64)

	c.w = core.NewVector()
}

func (c *SVM) Predict(sample *core.Sample) float64 {
	x := sample.GetFeatureVector()
	return c.PredictVector(x)
}

func (c *SVM) PredictVector(x *core.Vector) float64 {
	ret := c.w.Dot(x) - c.b
	return ret
}

func (c *SVM) MatchKKT(y, f, a float64) bool {
	ep := c.C * 0.01
	if a < ep && y*f > 1.0 {
		return true
	}

	if a > c.C-ep && y*f < 1.0 {
		return true
	}

	if a > ep && a < c.C-ep && y*f == 1.0 {
		return true
	}

	return false
}

func (c *SVM) Train(dataset *core.DataSet) {
	c.sv = []*core.Vector{}
	c.y = []float64{}
	c.a = []float64{}
	for k, sample := range dataset.Samples {
		x := sample.GetFeatureVector()
		c.sv = append(c.sv, x)
		c.xx = append(c.xx, x.Dot(x))
		if sample.Label > 0.0 {
			c.y = append(c.y, 1.0)
		} else {
			c.y = append(c.y, -1.0)
		}
		c.a = append(c.a, c.C*rand.Float64())
		c.w.AddVector(x, c.y[k]*c.a[k])
	}

	c.b = 0.0
	for k, x := range c.sv {
		c.b += c.PredictVector(x) - c.y[k]
	}
	c.b /= float64(len(c.sv))
	fmt.Println(c.b)

	for step := 0; step < 100; step++ {
		da := 0.0
		for i1 := 0; i1 < len(c.sv); i1++ {
			a1 := c.a[i1]
			x1 := c.sv[i1]
			y1 := c.y[i1]
			p1 := c.PredictVector(x1)
			if c.MatchKKT(y1, p1, a1) {
				continue
			}
			maxde := 0.0
			best_values := SVMValues{}
			for k2 := 0; k2 < 10; k2++ {
				i2 := rand.Intn(len(c.sv))
				if i1 == i2 {
					continue
				}

				x2 := c.sv[i2]
				y2 := c.y[i2]
				p2 := c.PredictVector(x2)
				k11 := c.xx[i1]
				k12 := x1.Dot(x2)
				k22 := c.xx[i2]

				a2 := c.a[i2]

				u := math.Max(0, a2-a1)
				v := math.Min(c.C, c.C+a2-a1)
				if y1*y2 > 0.0 {
					u = math.Max(0, a2+a1-c.C)
					v = math.Min(c.C, a1+a2)
				}

				e1 := p1 - y1
				e2 := p2 - y2

				a2old := a2
				a2 += y2 * (e1 - e2) / (k11 + k22 - 2*k12)

				a2 = math.Max(u, math.Min(a2, v))

				a1 += y1 * y2 * (a2old - a2)

				if math.Abs(e1-e2) > maxde {
					maxde = math.Abs(e1 - e2)
					best_values.a1 = a1
					best_values.a2 = a2
					best_values.i1 = i1
					best_values.i2 = i2
					best_values.e1 = e1
					best_values.e2 = e2
				}
				if maxde >= 4.0 {
					break
				}
			}
			da += math.Abs(c.a[best_values.i1] - best_values.a1)
			c.w.AddVector(c.sv[best_values.i1], c.y[best_values.i1]*(best_values.a1-c.a[best_values.i1]))
			c.w.AddVector(c.sv[best_values.i2], c.y[best_values.i2]*(best_values.a2-c.a[best_values.i2]))
			/*
				b1 := c.b - best_values.e1 - c.y[best_values.i1] * (best_values.a1 - c.a[best_values.i1]) * best_values.k11 - c.y[best_values.i2] * (best_values.a2 - c.a[best_values.i2]) * best_values.k12
				b2 := c.b - best_values.e2 - c.y[best_values.i1] * (best_values.a1 - c.a[best_values.i1]) * best_values.k12 - c.y[best_values.i2] * (best_values.a2 - c.a[best_values.i2]) * best_values.k22
				if best_values.a1 > 0.0 && best_values.a1 < c.C{
					c.b = b1
				} else {
					if best_values.a2 > 0.0 && best_values.a2 < c.C {
						c.b = b2
					} else {
						c.b = 0.5 * (b1 + b2)
					}
				}*/
			c.a[best_values.i1] = best_values.a1
			c.a[best_values.i2] = best_values.a2
		}
		da /= float64(len(c.sv))
		fmt.Printf(".. %f %f\n", da, c.b)
		if da < c.e {
			break
		}
	}
}
