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

func NewMinMaxScaler(dim int, groups [][]int) *MinMaxScaler {
	min := make([]float64, dim)
	max := make([]float64, dim)
	return &MinMaxScaler{
		Min:    min,
		Max:    max,
		Groups: groups,
	}
}

func (m *MinMaxScaler) Fit(data *Matrix) {
	for _, group := range m.Groups {
		min := math.Inf(1)
		max := math.Inf(-1)
		for i := range data.Rows {
			for _, col := range group {
				if data.Data[i*data.Cols+col] < min {
					min = data.Data[i*data.Cols+col]
				}
				if data.Data[i*data.Cols+col] > max {
					max = data.Data[i*data.Cols+col]
				}
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
	for _, group := range m.Groups {
		for i := range data.Rows {
			for _, col := range group {
				if diff := m.Max[col] - m.Min[col]; diff == 0 {
					result.Data[i*data.Cols+col] = 0
				} else {
					result.Data[i*data.Cols+col] = (data.Data[i*data.Cols+col] - m.Min[col]) / diff
				}
			}
		}
	}
	return result
}

type ZScoreScaler struct {
	Mean         []float64 `json:"mean"`
	StdDeviation []float64 `json:"stdDeviation"`
	Groups       [][]int   `json:"-"`
}

func NewZScoreScaler(dim int, groups [][]int) *ZScoreScaler {
	mean := make([]float64, dim)
	stdDeviation := make([]float64, dim)
	return &ZScoreScaler{
		Mean:         mean,
		StdDeviation: stdDeviation,
		Groups:       groups,
	}
}

func (m *ZScoreScaler) Fit(data *Matrix) {
	for _, group := range m.Groups {
		mean := 0.0
		for i := range data.Rows {
			for _, col := range group {
				mean += data.Data[i*data.Cols+col]
			}
		}
		mean /= float64(data.Rows * len(group))
		for _, col := range group {
			m.Mean[col] = mean
		}
	}
	for _, group := range m.Groups {
		stdDeviation := 0.0
		for i := range data.Rows {
			for _, col := range group {
				stdDeviation += math.Pow(data.Data[i*data.Cols+col]-m.Mean[col], 2)
			}
		}
		stdDeviation = math.Sqrt(stdDeviation / float64(data.Rows*len(group)))
		for _, col := range group {
			m.StdDeviation[col] = stdDeviation
		}
	}
}

func (m *ZScoreScaler) Transform(data *Matrix) *Matrix {
	result := NewConstMatrix(data.Rows, data.Cols, 0)
	for i, v := range data.Data {
		result.Data[i] = v
	}
	for _, group := range m.Groups {
		for i := range data.Rows {
			for _, col := range group {
				result.Data[i*data.Cols+col] = (data.Data[i*data.Cols+col] - m.Mean[col]) / m.StdDeviation[col]
			}
		}
	}
	return result
}
