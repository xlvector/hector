package hector 

// Description: function for minimizer such as LBFGS and OWLQN 
type DiffFunction interface {
    Value(pos *Vector) float64
    Gradient(pos *Vector) *Vector
}
