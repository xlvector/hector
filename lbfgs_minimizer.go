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
    fmt.Println("Iter\tcost\timprovement")
    fmt.Printf("%d\t%e\tUndefined\n", 0, cost)
    for iter:=1; iter <= m.maxIteration; iter++ {
        dir := grad.Copy()
        dir.ApplyScale(-1.0)
        helper.ApplyQuasiInverseHession(dir)
        newCost, newPos := helper.BackTrackingLineSearch(cost, pos, grad, dir, iter==1)
        if cost == newCost {
            break
        }
        cost = newCost
        pos = newPos
        terminalCriterion.addCost(cost)
        fmt.Printf("%d\t%e\t%e\n", iter, newCost, terminalCriterion.improvement)
        if terminalCriterion.isTerminable() {
            break
        }
        grad = costfun.Gradient(pos).Copy()
		if helper.UpdateState(pos, grad) {
			break
		}
    }
	return pos
}

func (m *LBFGSMinimizer) Evaluate(pos *Vector) float64 {
	return m.costFun.Value(pos)
}

func (m *LBFGSMinimizer) NextPoint(curPos *Vector, dir *Vector, alpha float64) *Vector {
	return curPos.ElemWiseMultiplyAdd(dir, alpha)
}
