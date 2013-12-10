package hector

import ("fmt"
        "math")

/**
 * It's based the paper "Scalable Training of L1-Regularized Log-Linear Models"
 * by Galen Andrew and Jianfeng Gao
 * user: weixuan
 * To change this template use File | Settings | File Templates.
 */
type OWLQNMinimizer struct {
    l1reg float64
	costFun DiffFunction
    numHist int
    maxIteration int
    tolerance float64
}

var owlqn_output_switch bool = false

func NewOWLQNMinimizer(l1reg float64) *OWLQNMinimizer {
    m := new(OWLQNMinimizer)
    m.l1reg = l1reg
    m.numHist = 10
    m.maxIteration = 20
    m.tolerance = 1e-4
    return m
}

func (m *OWLQNMinimizer) Minimize(costfun DiffFunction, init *Vector) *Vector {
	m.costFun = costfun;
    var cost float64 = m.Evaluate(init)
    var grad *Vector = costfun.Gradient(init).Copy()
    var pos *Vector = init.Copy()
    var terminalCriterion *relativeMeanImprCriterion = NewRelativeMeanImprCriterion(m.tolerance)
    terminalCriterion.addCost(cost)

    var helper *QuasiNewtonHelper = NewQuasiNewtonHelper(m.numHist, m, pos, grad)
    if owlqn_output_switch {
        fmt.Println("Iter\tcost\timprovement")
        fmt.Printf("%d\t%e\tUndefined", 0, cost)
    }
    for iter:=1; iter <= m.maxIteration; iter++ {
        // customed steepest descending dir
        steepestDescDir := grad.Copy()
        m.updateGrad(pos, steepestDescDir)
        steepestDescDir.ApplyScale(-1.0)
        dir := steepestDescDir.Copy()
        // quasi-newton dir
        helper.ApplyQuasiInverseHession(dir)
        m.fixDirSign(dir, steepestDescDir)
        // customed grad for the new position
        potentialGrad := grad.Copy()
        m.updateGradForNewPos(pos, potentialGrad, dir)
        newCost, newPos := helper.BackTrackingLineSearch(cost, pos, potentialGrad, dir, iter==1)
        if owlqn_output_switch {
            fmt.Println("")
        }
        if cost == newCost {
            break
        }
        cost = newCost
        pos = newPos
        grad = costfun.Gradient(pos).Copy()
        terminalCriterion.addCost(cost)
        if owlqn_output_switch {
            fmt.Printf("%d\t%e\t%e", iter, newCost, terminalCriterion.improvement)
        }
        if terminalCriterion.isTerminable() || helper.UpdateState(pos, grad) {
            if owlqn_output_switch {
                fmt.Println("")
            }
			break
		}
    }
	return pos
}

// Description: assume all the features in x also appears in grad
//              all the features in dir must be in grad
func (m *OWLQNMinimizer) updateGradForNewPos(x *Vector, grad *Vector, dir *Vector) {
    if m.l1reg == 0 {
        return
    }
    for key, val := range grad.data {
        xval := x.GetValue(key)
        if xval < 0 {
            grad.SetValue(key, val - m.l1reg)
        } else if xval > 0 {
            grad.SetValue(key, val + m.l1reg)
        } else {
            dirval := dir.GetValue(key)
            if dirval < 0 {
                grad.SetValue(key, val - m.l1reg)
            } else if dirval > 0 {
                grad.SetValue(key, val + m.l1reg)
            }
        }
    }
    return
}

// Description: assume all the features in x also appears in grad
func (m *OWLQNMinimizer) updateGrad(x *Vector, grad *Vector) {
    if m.l1reg == 0 {
        return
    }
    for key, val := range grad.data {
        xval := x.GetValue(key)
        if xval < 0 {
            grad.SetValue(key, val - m.l1reg)
        } else if xval > 0 {
            grad.SetValue(key, val + m.l1reg)
        } else {
            if val < -m.l1reg {
                grad.SetValue(key, val + m.l1reg)
            } else if val > m.l1reg {
                grad.SetValue(key, val - m.l1reg)
            }
        }
    }
    return
}

func (m *OWLQNMinimizer) fixDirSign(dir *Vector, steepestDescDir *Vector) {
    if m.l1reg == 0 {
        return
    }
    for key, val := range dir.data {
        if val * steepestDescDir.GetValue(key) <= 0 {
            dir.SetValue(key, 0)
        }
    }
}

func (m *OWLQNMinimizer) Evaluate(pos *Vector) float64 {
    cost := m.costFun.Value(pos)
    for _, val := range pos.data {
        cost += math.Abs(val) * m.l1reg
    }
    return cost
}

func (m *OWLQNMinimizer) NextPoint(curPos *Vector, dir *Vector, alpha float64) *Vector {
    if owlqn_output_switch {
        fmt.Printf(".")
    }
    newPos := curPos.ElemWiseMultiplyAdd(dir, alpha)
    if m.l1reg > 0 {
        for key, val := range curPos.data {
            if val * newPos.GetValue(key) < 0 {
                newPos.SetValue(key, 0)
            }
        }
    }
    return newPos
}
