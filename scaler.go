package goraph

import "math"

type Scaler interface {
	Fit(data *Matrix)
	Transform(data *Matrix) *Matrix
}

type MinMaxScaler struct {
	Min []float64 `json:"min"`
	Max []float64 `json:"max"`
}

func NewMinMaxScaler(n int) *MinMaxScaler {
	min := make([]float64, n)
	max := make([]float64, n)
	for i := range n {
		min[i] = math.Inf(1)
		max[i] = math.Inf(-1)
	}
	return &MinMaxScaler{
		Min: min,
		Max: max,
	}
}

func (m *MinMaxScaler) Fit(data *Matrix) {
	for i, v := range data.Data {
		if col := i % data.Cols; v < m.Min[col] {
			m.Min[col] = v
		} else if v > m.Max[col] {
			m.Max[col] = v
		}
	}
}

func (m *MinMaxScaler) Transform(data *Matrix) *Matrix {
	result := NewConstMatrix(data.Rows, data.Cols, 0)
	for i, v := range data.Data {
		col := i % data.Cols
		if diff := m.Max[col] - m.Min[col]; diff == 0 {
			result.Data[i] = 0
		} else {
			result.Data[i] = (v - m.Min[col]) / diff
		}
	}
	return result
}
