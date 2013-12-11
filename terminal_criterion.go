package hector 

import ("math")

/**
 * It's based the paper "Scalable Training of L1-Regularized Log-Linear Models"
 * by Galen Andrew and Jianfeng Gao
 * user: weixuan
 * To change this template use File | Settings | File Templates.
 */
type relativeMeanImprCriterion struct {
    minHist int
    maxHist int
    tolerance float64
    improvement float64
    costList []float64
}

func NewRelativeMeanImprCriterion(tolerance float64) *relativeMeanImprCriterion {
    tc := new(relativeMeanImprCriterion)
    tc.minHist = 5
    tc.maxHist = 10
    tc.costList = make([]float64, 0, tc.maxHist)
    tc.tolerance = tolerance
    return tc
}

func (tc *relativeMeanImprCriterion) calImprovement() float64{
    sz := len(tc.costList)
    if sz <= tc.minHist {
        return math.MaxFloat32
    }
    first := tc.costList[0]
    last := tc.costList[sz-1]
    impr := (first - last) /float64(sz-1)
    if last != 0 {
        impr = math.Abs(impr / last)
    } else if first != 0 {
        impr = math.Abs(impr / first)
    } else {
        impr = 0
    }
    if sz > tc.maxHist {
        tc.costList = tc.costList[1:]
    }
    return impr
}

func (tc *relativeMeanImprCriterion) addCost(latestCost float64) {
    tc.costList = append(tc.costList, latestCost)
    tc.improvement = tc.calImprovement()
}

func (tc *relativeMeanImprCriterion) isTerminable() bool {
    return tc.improvement  <= tc.tolerance
}
