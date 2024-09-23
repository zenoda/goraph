package goraph

import "math"

type Scaler interface {
	Fit(data [][]float64)
	Transform(data [][]float64) [][]float64
}

type MinMaxScaler struct {
	Min    []float64 `json:"min"`
	Max    []float64 `json:"max"`
	Groups [][]int   `json:"groups"`
	Dim    int       `json:"dim"`
}

func NewMinMaxScaler(dim int, groups [][]int) *MinMaxScaler {
	minData := make([]float64, len(groups))
	maxData := make([]float64, len(groups))
	return &MinMaxScaler{
		Min:    minData,
		Max:    maxData,
		Groups: groups,
		Dim:    dim,
	}
}

func (m *MinMaxScaler) Fit(data [][]float64) {
	for i, group := range m.Groups {
		minVal := math.Inf(1)
		maxVal := math.Inf(-1)
		for _, item := range data {
			for j := range len(item) / m.Dim {
				for _, col := range group {
					minVal = min(minVal, item[j*m.Dim+col])
					maxVal = max(maxVal, item[j*m.Dim+col])
				}
			}
		}
		m.Min[i] = minVal
		m.Max[i] = maxVal
	}
}

func (m *MinMaxScaler) Transform(data [][]float64) [][]float64 {
	result := make([][]float64, len(data))
	for i := range result {
		result[i] = make([]float64, len(data[i]))
		copy(result[i], data[i])
	}
	for i, group := range m.Groups {
		for p, item := range data {
			for j := range len(item) / m.Dim {
				for _, col := range group {
					result[p][j*m.Dim+col] = (item[j*m.Dim+col] - m.Min[i]) / (m.Max[i] - m.Min[i])
				}
			}
		}
	}
	return result
}

//
//type ZScoreScaler struct {
//	Mean         []float64 `json:"mean"`
//	StdDeviation []float64 `json:"stdDeviation"`
//	Groups       [][]int   `json:"-"`
//}
//
//func NewZScoreScaler(dim int, groups [][]int) *ZScoreScaler {
//	mean := make([]float64, dim)
//	stdDeviation := make([]float64, dim)
//	return &ZScoreScaler{
//		Mean:         mean,
//		StdDeviation: stdDeviation,
//		Groups:       groups,
//	}
//}
//
//func (m *ZScoreScaler) Fit(data *Matrix) {
//	for _, group := range m.Groups {
//		mean := 0.0
//		for i := range data.Rows {
//			for _, col := range group {
//				mean += data.Data[i*data.Cols+col]
//			}
//		}
//		mean /= float64(data.Rows * len(group))
//		for _, col := range group {
//			m.Mean[col] = mean
//		}
//	}
//	for _, group := range m.Groups {
//		stdDeviation := 0.0
//		for i := range data.Rows {
//			for _, col := range group {
//				stdDeviation += math.Pow(data.Data[i*data.Cols+col]-m.Mean[col], 2)
//			}
//		}
//		stdDeviation = math.Sqrt(stdDeviation / float64(data.Rows*len(group)))
//		for _, col := range group {
//			m.StdDeviation[col] = stdDeviation
//		}
//	}
//}
//
//func (m *ZScoreScaler) Transform(data *Matrix) *Matrix {
//	result := NewConstMatrix(data.Rows, data.Cols, 0)
//	for i, v := range data.Data {
//		result.Data[i] = v
//	}
//	for _, group := range m.Groups {
//		for i := range data.Rows {
//			for _, col := range group {
//				result.Data[i*data.Cols+col] = (data.Data[i*data.Cols+col] - m.Mean[col]) / m.StdDeviation[col]
//			}
//		}
//	}
//	return result
//}
