package gp

import (
	"github.com/xlvector/hector/core"
	"math"
	"strconv"
)

type GaussianProcessParameters struct {
	Dim   int64
	Theta float64
}

type GaussianProcess struct {
	Params            GaussianProcessParameters
	CovarianceFunc    CovFunc
	CovMatrix         *core.Matrix
	TargetValues      *core.Vector
	InvCovTarget      *core.Vector // inv(CovMatrix)*TargetValues
	DataSet           *core.RealDataSet
	TrainingDataCount int64
}

func (self *GaussianProcess) SaveModel(path string) {

}

func (self *GaussianProcess) LoadModel(path string) {

}

/*
   Given matrix m and vector v, compute inv(m)*v.
   Based on Gibbs and MacKay 1997, and Mark N. Gibbs's PhD dissertation

   Details:
   A - positive seminidefinite matrix
   u - a vector
   theta - positive number
   C = A + I*theta
   Returns inv(C)*u - So you need the diagonal noise term for covariance matrix in a sense.
   However, this algorithm is numerically stable, the noise term can be very small and the inversion can still be calculated...
*/
func (algo *GaussianProcess) ApproximateInversion(A *core.Matrix, u *core.Vector, theta float64, dim int64) *core.Vector {
	max_itr := 500
	tol := 0.01

	C := core.NewMatrix()
	for key, val := range A.Data {
		C.Data[key] = val.Copy()
	}

	// Add theta to diagonal elements
	for i := int64(0); i < dim; i++ {
		_, ok := C.Data[i]
		if !ok {
			C.Data[i] = core.NewVector()
		}
		C.Data[i].Data[i] = C.Data[i].Data[i] + theta
	}

	var Q_l float64
	var Q_u float64
	var dQ float64
	u_norm := u.Dot(u) / 2

	// Lower bound
	y_l := core.NewVector()
	g_l := u.Copy()
	h_l := u.Copy()
	lambda_l := float64(0)
	gamma_l := float64(0)
	var tmp_f1 float64
	var tmp_f2 float64
	var tmp_v1 *core.Vector
	tmp_f1 = g_l.Dot(g_l)
	tmp_v1 = C.MultiplyVector(h_l)

	// Upper bound
	y_u := core.NewVector()
	g_u := u.Copy()
	h_u := u.Copy()
	lambda_u := float64(0)
	gamma_u := float64(0)
	var tmp_f3 float64
	var tmp_f4 float64
	var tmp_v3 *core.Vector
	var tmp_v4 *core.Vector
	tmp_v3 = g_u.MultiplyMatrix(A)
	tmp_v4 = C.MultiplyVector(h_u)
	tmp_f3 = tmp_v1.Dot(g_u)

	for i := 0; i < max_itr; i++ {
		// Lower bound
		lambda_l = tmp_f1 / h_l.Dot(tmp_v1)
		y_l.AddVector(h_l, lambda_l) //y_l next
		Q_l = y_l.Dot(u) - 0.5*(y_l.MultiplyMatrix(C)).Dot(y_l)

		// Upper bound
		lambda_u = tmp_f3 / tmp_v3.Dot(tmp_v4)
		y_u.AddVector(h_u, lambda_u) //y_u next
		Q_u = (y_u.MultiplyMatrix(A)).Dot(u) - 0.5*((y_u.MultiplyMatrix(C)).MultiplyMatrix(A)).Dot(y_u)

		dQ = (u_norm-Q_u)/theta - Q_l
		if dQ < tol {
			break
		}

		// Lower bound var updates
		g_l.AddVector(tmp_v1, -lambda_l) //g_l next
		tmp_f2 = g_l.Dot(g_l)
		gamma_l = tmp_f2 / tmp_f1
		for key, val := range h_l.Data {
			h_l.SetValue(key, val*gamma_l)
		}
		h_l.AddVector(g_l, 1)          //h_l next
		tmp_f1 = tmp_f2                //tmp_f1 next
		tmp_v1 = C.MultiplyVector(h_l) //tmp_v1 next

		// Upper bound var updates
		g_u.AddVector(tmp_v4, -lambda_u) //g_u next
		tmp_v3 = g_u.MultiplyMatrix(A)   //tmp_v3 next
		tmp_f4 = tmp_v3.Dot(g_u)
		gamma_u = tmp_f4 / tmp_f3
		for key, val := range h_u.Data {
			h_u.SetValue(key, val*gamma_u)
		}
		h_u.AddVector(g_u, 1)          //h_u next
		tmp_v4 = C.MultiplyVector(h_u) //tmp_v4 next
		tmp_f3 = tmp_f4                // tmp_f3 next
	}

	return y_l
}

func (algo *GaussianProcess) ExtractTargetValuesAsVector(samples []*core.RealSample) *core.Vector {
	targets := core.NewVector()
	for i := 0; i < len(samples); i++ {
		targets.SetValue(int64(i), samples[i].Value)
	}
	return targets
}

func (algo *GaussianProcess) Init(params map[string]string) {

	dim, _ := strconv.ParseInt(params["dim"], 10, 64)

	algo.Params = GaussianProcessParameters{}
	algo.Params.Dim = dim    // Pass in dim as a param.. and require feature space to be continous.
	algo.Params.Theta = 1e-7 // Used by approximate inversion as the diagonal noise

	radius := 0.1
	camp := 40.0
	cf := CovSEARD{}
	radiuses := core.NewVector()
	for i := int64(1); i <= dim; i++ {
		radiuses.SetValue(i, radius)
	}
	cf.Init(radiuses, camp)

	algo.CovarianceFunc = cf.Cov
}

func (algo *GaussianProcess) Train(dataset *core.RealDataSet) {
	algo.DataSet = dataset
	algo.TrainingDataCount = int64(len(dataset.Samples))
	algo.CovMatrix = CovMatrix(algo.DataSet.Samples, algo.CovarianceFunc)
	algo.TargetValues = algo.ExtractTargetValuesAsVector(algo.DataSet.Samples)
	algo.InvCovTarget = algo.ApproximateInversion(algo.CovMatrix, algo.TargetValues, algo.Params.Theta, algo.TrainingDataCount)
}

func (algo *GaussianProcess) Predict(sample *core.RealSample) float64 {
	k := CovVector(algo.DataSet.Samples, sample, algo.CovarianceFunc)
	pred := k.Dot(algo.InvCovTarget)

	return pred
}

func (algo *GaussianProcess) PredictStd(sample *core.RealSample) float64 {
	k := CovVector(algo.DataSet.Samples, sample, algo.CovarianceFunc)
	C_inv_k := algo.ApproximateInversion(algo.CovMatrix, k, algo.Params.Theta, algo.TrainingDataCount)
	std := math.Sqrt(algo.CovarianceFunc(sample.GetFeatureVector(), sample.GetFeatureVector()) - k.Dot(C_inv_k))
	return std
}
