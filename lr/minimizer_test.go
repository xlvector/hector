package lr

import (
	"github.com/xlvector/hector/core"
	"math"
	"testing"
)

type mseDiffFunction struct {
	center  core.Vector
	weights core.Vector
	grad    core.Vector
	init    core.Vector
}

func getMSECostFunction() *mseDiffFunction {
	f := new(mseDiffFunction)
	f.center.Data = map[int64]float64{}
	f.weights.Data = map[int64]float64{0: 1, 1: 0.01}
	f.init.Data = map[int64]float64{0: 1, 1: 1}
	f.grad.Data = map[int64]float64{0: 0, 1: 0}
	return f
}

func (f *mseDiffFunction) Value(x *core.Vector) float64 {
	var cost float64 = 0
	for n, val := range x.Data {
		diff := val - f.center.GetValue(n)
		cost += f.weights.GetValue(n) * diff * diff
	}
	return 0.5 * cost
}

// Gradients for different points could use the same memory
func (f *mseDiffFunction) Gradient(x *core.Vector) *core.Vector {
	for n, val := range x.Data {
		f.grad.SetValue(n, f.weights.GetValue(n)*(val-f.center.GetValue(n)))
	}
	return &f.grad
}

func (f *mseDiffFunction) testResult(result *core.Vector, tolerance float64, t *testing.T) {
	for n, val := range result.Data {
		if math.Abs(val-f.center.GetValue(n)) > tolerance {
			t.Errorf("Mismatch\nIndex\tTrue\tResult\n%d\t%e\t%e\n", n, f.center.GetValue(n), val)
		}
	}
}

func TestLBFGS(t *testing.T) {
	diffFunc := getMSECostFunction()
	minimizer := NewLBFGSMinimizer()
	result := minimizer.Minimize(diffFunc, &(diffFunc.init))
	diffFunc.testResult(result, 1e-6, t)
}

func TestOWLQN(t *testing.T) {
	diffFunc := getMSECostFunction()
	minimizer := NewOWLQNMinimizer(0.001)
	result := minimizer.Minimize(diffFunc, &(diffFunc.init))
	diffFunc.testResult(result, 0, t)
}
