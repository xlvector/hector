package hector

import (
    "math"
)

/**
 * It's based the paper "Scalable Training of L1-Regularized Log-Linear Models"
 * by Galen Andrew and Jianfeng Gao
 * user: weixuan
 * To change this template use File | Settings | File Templates.
 */
type QuasiNewtonHelper struct {
    // config
    numHist int64
    minimizer Minimizer
    // historical data
    sList, yList []*Vector
    roList []float64
	curPos, curGrad *Vector
}

type Minimizer interface {
    NextPoint(curPos *Vector, dir *Vector, alpha float64) *Vector
	Evaluate(curPos *Vector) float64
}

const MAX_BACKTRACKING_ITER = 50

// Description: the pos and gradient arguments should NOT be modified outside
func NewQuasiNewtonHelper(numHist int, minimizer Minimizer, curPos *Vector, curGrad *Vector) (*QuasiNewtonHelper) {
    h := new(QuasiNewtonHelper)
    h.numHist = int64(numHist)
	h.minimizer = minimizer
	h.curPos = curPos
	h.curGrad = curGrad
    h.sList = make([]*Vector, 0)
    h.yList = make([]*Vector, 0)
    h.roList = make([]float64, 0)
    return h
}

// Description: Update the dir from -grad to optimal direction
//              Dir will be modified directly
func (h *QuasiNewtonHelper) ApplyQuasiInverseHession(dir *Vector) {
    count := len(h.sList)
    if count == 0 {
        return
    }
    alphas := make([]float64, count, count)
    for n:=count-1; n>=0; n-- {
        alphas[n] = -dir.Dot(h.sList[n]) / h.roList[n]
        dir.ApplyElemWiseMultiplyAccumulation(h.yList[n], alphas[n])
    }
    lastY := h.yList[count-1]
    yDotY := lastY.Dot(lastY)
    scalar := h.roList[count-1] / yDotY
    dir.ApplyScale(scalar)

    for n:=0; n<count; n++ {
        beta := dir.Dot(h.yList[n]) / h.roList[n]
        dir.ApplyElemWiseMultiplyAccumulation(h.sList[n], -alphas[n] - beta)
    }
	return
}

func (h *QuasiNewtonHelper) BackTrackingLineSearch(cost float64, pos *Vector, grad *Vector, dir *Vector, isInit bool) (nextCost float64, nextPos *Vector) {
    dotGradDir := grad.Dot(dir)
	if dotGradDir == 0 {
		return cost, pos
	}
	if dotGradDir > 0 {
		panic("BackTracking: to the opposite direction of grad")
	}

    alpha := 1.0
    backoff := 0.5
    if isInit {
        normDir := math.Sqrt(dir.Dot(dir))
        alpha = (1/normDir)
        backoff = 0.1
    }

    var c1 float64 = 1e-4
    for cntItr:=0; cntItr <= MAX_BACKTRACKING_ITER; cntItr++ {
        nextPos = h.minimizer.NextPoint(pos, dir, alpha)
        nextCost = h.minimizer.Evaluate(nextPos)
        if (nextCost <= cost + c1 * dotGradDir * alpha) {
            break
		}
        alpha *= backoff
    }
    return nextCost, nextPos
}

// Description: the pos and gradient arguments should NOT be modified outside
func (h *QuasiNewtonHelper) UpdateState(nextPos *Vector, nextGrad *Vector) (isOptimal bool) {
	if int64(len(h.sList)) >= h.numHist {
		h.sList = h.sList[1:]
		h.yList = h.yList[1:]
		h.roList = h.roList[1:]
	}
    newS := nextPos.ElemWiseMultiplyAdd(h.curPos, -1)
	newY := nextGrad.ElemWiseMultiplyAdd(h.curGrad, -1)	
	ro := newS.Dot(newY)
	h.sList = append(h.sList, newS)
	h.yList = append(h.yList, newY)
	h.roList = append(h.roList, ro)
	h.curPos = nextPos
	h.curGrad = nextGrad
	return ro == 0
}
