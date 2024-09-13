package goraph

import "math"

type Scaler interface {
	Fit(data *Matrix)
	Transform(data *Matrix) *Matrix
}

type MinMaxScaler struct {
	Min    []float64 `json:"min"`
	Max    []float64 `json:"max"`
	Groups [][]int   `json:"groups"`
}

func NewMinMaxScaler(n int, groups [][]int) *MinMaxScaler {
	min := make([]float64, n)
	max := make([]float64, n)
	for i := range n {
		min[i] = math.Inf(1)
		max[i] = math.Inf(-1)
	}
	return &MinMaxScaler{
		Min:    min,
		Max:    max,
		Groups: groups,
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
	for _, group := range m.Groups {
		min := math.Inf(1)
		max := math.Inf(-1)
		for _, col := range group {
			if m.Min[col] < min {
				min = m.Min[col]
			}
			if m.Max[col] > max {
				max = m.Max[col]
			}
		}
		for _, col := range group {
			m.Min[col] = min
			m.Max[col] = max
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
