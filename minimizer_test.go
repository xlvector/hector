package hector

import(
    "fmt"
    "testing"
    "math"
)

type mseDiffFunction struct {
    center Vector
    weights Vector
    grad Vector
    init Vector
}

func getMSECostFunction() *mseDiffFunction{
	f := new(mseDiffFunction)
    f.center.data = map[int64]float64 {0:0, 1:0}
    f.weights.data = map[int64]float64 {0:1, 1:0.01}
    f.init.data = map[int64]float64 {0:1, 1:1}
    f.grad.data = map[int64]float64 {0:0, 1:0}
    return f
}

func (f *mseDiffFunction) Value(x *Vector) float64 {
    var cost float64 = 0
    for n, val := range x.data {
		diff := val - f.center.GetValue(n)
        cost += f.weights.GetValue(n) * diff * diff
    }
    return 0.5 * cost
}

// Gradients for different points could use the same memory
func (f *mseDiffFunction) Gradient(x *Vector) *Vector {
    for n, val := range x.data {
        f.grad.SetValue(n, f.weights.GetValue(n) * (val - f.center.GetValue(n)))
    }
    return &f.grad
}

func (f *mseDiffFunction) testResult(result *Vector, tolerance float64, t *testing.T) {
	fmt.Println("Index\tTrue\tResult")
    for n, val := range f.center.data {
		fmt.Printf("%d\t%e\t%e\n", n, val, result.GetValue(n))
	}
    for n, val := range result.data {
		if math.Abs(val - f.center.GetValue(n)) > tolerance {
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
