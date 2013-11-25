package hector

import ("fmt")

const numHist int = 10
const maxIteration int = 200

type LBFGSMinimizer struct {
	costFun DiffFunction
}

type DiffFunction interface {
    Value(pos *Vector) float64
    Gradient(pos *Vector) *Vector
}

func (minimizer *LBFGSMinimizer) Minimize(costfun DiffFunction, init *Vector) *Vector {
	minimizer.costFun = costfun;
    var cost float64 = costfun.Value(init)
    var grad *Vector = costfun.Gradient(init).Copy()
    var pos *Vector = init

    var helper *QuasiNewtonHelper = NewQuasiNewtonHelper(numHist, minimizer, pos, grad)
    fmt.Println("Iter\tcost\timprovement")
    fmt.Printf("%d\t%eN/A\n", 0, cost)
    for iter:=1; iter <= maxIteration; iter++ {
        dir := grad.Copy()
        dir.ApplyScale(-1.0)
        helper.ApplyQuasiInverseHession(dir)
        newCost, newPos := helper.BackTrackingLineSearch(cost, pos, grad, dir, iter==1)
        if cost <= newCost {
            break
        }
        fmt.Printf("%d\t%e\t%e\n", iter, newCost, (cost-newCost)/cost)
        if (cost-newCost)/cost <= 0.0001 {
            break
        }
        cost = newCost
        pos = newPos
        grad = costfun.Gradient(pos).Copy()
    }
	return pos
}

func (m *LBFGSMinimizer) Evaluate(pos *Vector) float64 {
	return m.costFun.Value(pos)
}

func (m *LBFGSMinimizer) NextPoint(curPos *Vector, dir *Vector, alpha float64) *Vector {
    return curPos.ElemWiseMultiplyAdd(dir, alpha)
}