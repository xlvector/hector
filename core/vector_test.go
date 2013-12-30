package core

import (
	"testing"
	"math"
)

func TestArrayVector(t *testing.T){
	a := NewArrayVector()
	precision := 1e-9

	a.AddValue(3, 1.78)

	if math.Abs(a.GetValue(3) - 1.78) > precision {
		t.Error("Get wrong value after set value")
	}

	a.AddValue(3, -1.1)

	if math.Abs(a.GetValue(3) - 0.68) > precision {
		t.Error("Add value wrong")
	}

	a.Scale(0.5)

	if math.Abs(a.GetValue(3) - 0.34) > precision {
		t.Error("Scale wrong")
	}
}