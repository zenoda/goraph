package goraph

import (
	"math"
	"sort"
)

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

type RobustScaler struct {
	Median []float64 `json:"median"`
	IQR    []float64 `json:"IQR"`
	Groups [][]int   `json:"groups"`
	Dim    int       `json:"dim"`
}

func NewRobustScaler(dim int, groups [][]int) *RobustScaler {
	return &RobustScaler{
		Groups: groups,
		Dim:    dim,
		Median: make([]float64, len(groups)),
		IQR:    make([]float64, len(groups)),
	}
}
func (m *RobustScaler) Fit(data [][]float64) {
	groupedData := make([][]float64, len(m.Groups))
	for i, group := range m.Groups {
		for _, item := range data {
			for row := range len(item) / m.Dim {
				for _, col := range group {
					groupedData[i] = append(groupedData[i], item[row*m.Dim+col])
				}
			}
		}
	}
	for i := range groupedData {
		sort.Float64s(groupedData[i])
		m.Median[i] = groupedData[i][int(float64(len(groupedData[i]))*0.5)]
		m.IQR[i] = groupedData[i][int(float64(len(groupedData[i]))*0.75)] - groupedData[i][int(float64(len(groupedData[i]))*0.25)]
	}
}
func (m *RobustScaler) Transform(data [][]float64) [][]float64 {
	result := make([][]float64, len(data))
	for i := range result {
		result[i] = make([]float64, len(data[i]))
		copy(result[i], data[i])
	}
	for i, group := range m.Groups {
		for idx, item := range data {
			for row := range len(item) / m.Dim {
				for _, col := range group {
					result[idx][row*m.Dim+col] = (data[idx][row*m.Dim+col] - m.Median[i]) / m.IQR[i]
				}
			}
		}
	}
	return result
}

type ZScoreScaler struct {
	Mean         []float64 `json:"mean"`
	StdDeviation []float64 `json:"stdDeviation"`
	Groups       [][]int   `json:"groups"`
	Dim          int       `json:"dim"`
}

func NewZScoreScaler(dim int, groups [][]int) *ZScoreScaler {
	return &ZScoreScaler{
		Mean:         make([]float64, len(groups)),
		StdDeviation: make([]float64, len(groups)),
		Groups:       groups,
		Dim:          dim,
	}
}

func (m *ZScoreScaler) Fit(data [][]float64) {
	groupedData := make([][]float64, len(m.Groups))
	for i, group := range m.Groups {
		for _, item := range data {
			for row := range len(item) / m.Dim {
				for _, col := range group {
					groupedData[i] = append(groupedData[i], item[row*m.Dim+col])
				}
			}
		}
	}
	for i := range groupedData {
		mean := 0.0
		for j := range groupedData[i] {
			mean += groupedData[i][j]
		}
		mean /= float64(len(groupedData[i]))
		stdDeviation := 0.0
		for j := range groupedData[i] {
			stdDeviation += math.Pow(groupedData[i][j]-mean, 2)
		}
		stdDeviation = math.Sqrt(stdDeviation / float64(len(groupedData[i])))
		m.Mean[i] = mean
		m.StdDeviation[i] = stdDeviation
	}
}

func (m *ZScoreScaler) Transform(data [][]float64) [][]float64 {
	result := make([][]float64, len(data))
	for i, item := range data {
		result[i] = make([]float64, len(item))
		copy(result[i], item)
	}
	for i, group := range m.Groups {
		for j, item := range data {
			for row := range len(item) / m.Dim {
				for _, col := range group {
					result[j][row*m.Dim+col] = (data[j][row*m.Dim+col] - m.Mean[i]) / m.StdDeviation[i]
				}
			}
		}
	}
	return result
}
