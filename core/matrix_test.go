package core

import (
	"testing"
	"math"
)

func TestMatrix(t *testing.T){
	a := NewMatrix()
	precision := 1e-9

	a.AddValue(3, 4, 1.78)

	if math.Abs(a.GetValue(3, 4) - 1.78) > precision {
		t.Error("Get wrong value after set value")
	}

	a.AddValue(3, 4, -1.1)

	if math.Abs(a.GetValue(3, 4) - 0.68) > precision {
		t.Error("Add value wrong")
	}

	b := NewMatrix()

	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			b.SetValue(int64(i), int64(j), 1.0)
		}
	}

	c := b.Scale(2.0)

	if math.Abs(c.GetValue(7,8) - 2.0) > precision {
		t.Error("scale function error")
	}
}