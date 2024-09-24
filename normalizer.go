package goraph

import "math"

type Normalizer interface {
	Normalize([][]float64) [][]float64
}

type L2Normalizer struct {
	Dim    int     `json:"dim"`
	Groups [][]int `json:"groups"`
}

func NewL2Normalizer(dim int, groups [][]int) *L2Normalizer {
	return &L2Normalizer{
		Dim:    dim,
		Groups: groups,
	}
}
func (n *L2Normalizer) Normalize(data [][]float64) [][]float64 {
	result := make([][]float64, len(data))
	for i, item := range data {
		result[i] = make([]float64, len(item))
		copy(result[i], item)
	}
	for i, item := range data {
		for rowIdx := range len(item) / n.Dim {
			for _, group := range n.Groups {
				mod := 0.0
				for _, colIdx := range group {
					mod += math.Pow(item[rowIdx*n.Dim+colIdx], 2)
				}
				mod = math.Sqrt(mod)
				for _, colIdx := range group {
					result[i][rowIdx*n.Dim+colIdx] = data[i][rowIdx*n.Dim+colIdx] / mod
				}
			}
		}
	}
	return result
}
