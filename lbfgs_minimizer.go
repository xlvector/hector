package hector

import ("fmt")

/**
 * It's based the paper "Scalable Training of L1-Regularized Log-Linear Models"
 * by Galen Andrew and Jianfeng Gao
 * user: weixuan
 * To change this template use File | Settings | File Templates.
 */
type LBFGSMinimizer struct {
	costFun DiffFunction
    numHist int
    maxIteration int
    tolerance float64
}

var lbfgs_output_switch bool = false

func NewLBFGSMinimizer() (*LBFGSMinimizer) {
    m := new(LBFGSMinimizer)
    m.numHist = 10
    m.maxIteration = 200
    m.tolerance = 1e-4
    return m
}

func (m *LBFGSMinimizer) Minimize(costfun DiffFunction, init *Vector) *Vector {
	m.costFun = costfun;
    var cost float64 = costfun.Value(init)
    var grad *Vector = costfun.Gradient(init).Copy()
    var pos *Vector = init.Copy()
    var terminalCriterion *relativeMeanImprCriterion  = NewRelativeMeanImprCriterion(m.tolerance)
    terminalCriterion.addCost(cost)

    var helper *QuasiNewtonHelper = NewQuasiNewtonHelper(m.numHist, m, pos, grad)
    if lbfgs_output_switch {
        fmt.Println("Iter\tcost\timprovement")
        fmt.Printf("%d\t%e\tUndefined", 0, cost)
    }
    for iter:=1; iter <= m.maxIteration; iter++ {
        dir := grad.Copy()
        dir.ApplyScale(-1.0)
        helper.ApplyQuasiInverseHession(dir)
        newCost, newPos := helper.BackTrackingLineSearch(cost, pos, grad, dir, iter==1)
        if lbfgs_output_switch {
            fmt.Println("")
        }
        if cost == newCost {
            break
        }
        cost = newCost
        pos = newPos
        grad = costfun.Gradient(pos).Copy()
        terminalCriterion.addCost(cost)
        if lbfgs_output_switch {
            fmt.Printf("%d\t%e\t%e", iter, newCost, terminalCriterion.improvement)
        }
        if terminalCriterion.isTerminable() || helper.UpdateState(pos, grad) {
            if lbfgs_output_switch {
                fmt.Println("")
            }
            break
		}
    }
	return pos
}

func (m *LBFGSMinimizer) Evaluate(pos *Vector) float64 {
	return m.costFun.Value(pos)
}

func (m *LBFGSMinimizer) NextPoint(curPos *Vector, dir *Vector, alpha float64) *Vector {
	if lbfgs_output_switch {
        fmt.Printf(".")
    }
    return curPos.ElemWiseMultiplyAdd(dir, alpha)
}
