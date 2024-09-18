package goraph

import (
	"math"
)

type Normalizer interface {
	Normalize(x *Matrix) *Matrix
}

type L2Normalizer struct {
	Groups [][]int `json:"-"`
}

func NewL2Normalizer(groups [][]int) *L2Normalizer {
	return &L2Normalizer{
		Groups: groups,
	}
}
func (n *L2Normalizer) Normalize(x *Matrix) *Matrix {
	result := NewConstMatrix(x.Rows, x.Cols, 0)
	for i, v := range x.Data {
		result.Data[i] = v
	}
	for _, colGroup := range n.Groups {
		for i := range x.Rows {
			norm := 0.0
			for _, col := range colGroup {
				norm += math.Pow(x.Data[i*x.Cols+col], 2)
			}
			norm = math.Sqrt(norm)
			for _, col := range colGroup {
				result.Data[i*x.Cols+col] = x.Data[i*x.Cols+col] / norm
			}
		}
	}
	return result
}
