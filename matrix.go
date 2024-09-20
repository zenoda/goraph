package goraph

import "fmt"

type Matrix struct {
	Data []float64 `json:"data"`
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}

func NewMatrix(rows, cols int, data []float64) *Matrix {
	if len(data) != rows*cols {
		panic("Data length does not match matrix dimensions")
	}
	return &Matrix{
		Data: data,
		Rows: rows,
		Cols: cols,
	}
}

func (m *Matrix) String() string {
	str := fmt.Sprintf("Shape:[%d,%d], Data:[", m.Rows, m.Cols)
	for i := range m.Rows {
		str += fmt.Sprintf("%v", m.Data[i*m.Cols:i*m.Cols+m.Cols])
	}
	str += fmt.Sprintf("]")
	return str
}

func NewConstMatrix(rows, cols int, value float64) *Matrix {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = value
	}
	return &Matrix{
		Data: data,
		Rows: rows,
		Cols: cols,
	}
}

func NewRandomMatrix(rows, cols int, f func() float64) *Matrix {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = f()
	}
	return &Matrix{
		Data: data,
		Rows: rows,
		Cols: cols,
	}
}

func (m *Matrix) Add(other *Matrix) (result *Matrix) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Matrix dimensions do not match")
	}
	data := make([]float64, m.Rows*m.Cols)
	for i := range data {
		data[i] = m.Data[i] + other.Data[i]
	}
	return NewMatrix(m.Rows, m.Cols, data)
}

func (m *Matrix) Multi(other *Matrix) (result *Matrix) {
	if m.Cols != other.Rows {
		panic("Matrix dimensions do not match")
	}
	data := make([]float64, m.Rows*other.Cols)
	for r1 := range m.Rows {
		for c2 := range other.Cols {
			for c1 := range m.Cols {
				data[r1*other.Cols+c2] += m.Data[r1*m.Cols+c1] * other.Data[c1*other.Cols+c2]
			}
		}
	}
	return NewMatrix(m.Rows, other.Cols, data)
}

func (m *Matrix) Sub(other *Matrix) (result *Matrix) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Matrix dimensions do not match")
	}
	data := make([]float64, m.Rows*m.Cols)
	for i := range data {
		data[i] = m.Data[i] - other.Data[i]
	}
	return NewMatrix(m.Rows, m.Cols, data)
}

func (m *Matrix) Negate() (result *Matrix) {
	data := make([]float64, m.Rows*m.Cols)
	for i := range data {
		data[i] = -m.Data[i]
	}
	return NewMatrix(m.Rows, m.Cols, data)
}

func (m *Matrix) Trans() (result *Matrix) {
	data := make([]float64, m.Rows*m.Cols)
	for i := range m.Rows {
		for j := range m.Cols {
			data[j*m.Rows+i] = m.Data[i*m.Cols+j]
		}
	}
	return NewMatrix(m.Cols, m.Rows, data)
}

func (m *Matrix) Scale(rate float64) *Matrix {
	data := make([]float64, m.Rows*m.Cols)
	for i := range data {
		data[i] = m.Data[i] * rate
	}
	return NewMatrix(m.Rows, m.Cols, data)
}

func (m *Matrix) HConcat(other *Matrix) (result *Matrix) {
	if m.Rows != other.Rows {
		panic("Matrix rows do not match")
	}
	var data []float64
	for i := range m.Rows {
		data = append(data, m.Data[i*m.Cols:i*m.Cols+m.Cols]...)
		data = append(data, other.Data[i*other.Cols:i*other.Cols+other.Cols]...)
	}
	return NewMatrix(m.Rows, m.Cols+other.Cols, data)
}

func (m *Matrix) VConcat(other *Matrix) (result *Matrix) {
	if m.Cols != other.Cols {
		panic("Matrix cols do not match")
	}
	var data []float64
	data = append(data, m.Data...)
	data = append(data, other.Data...)
	return NewMatrix(m.Rows+other.Rows, m.Cols, data)
}

func (m *Matrix) RowSlice(start, end int) *Matrix {
	if start > end {
		panic("start must be less than end")
	}
	if start >= m.Rows {
		panic("start must be less than the number of rows")
	}
	if end > m.Rows {
		panic("end must be less than or equal to the number of rows")
	}
	data := make([]float64, (end-start)*m.Cols)
	for i := range data {
		data[i] = m.Data[i+m.Cols*start]
	}
	return NewMatrix(end-start, m.Cols, data)
}

func (m *Matrix) ColSlice(start, end int) *Matrix {
	if start < end {
		panic("start must be greater than end")
	}
	if start >= m.Cols {
		panic("start must be less than the number of cols")
	}
	if end > m.Cols {
		panic("end must be less than or equal to the number of cols")
	}
	data := make([]float64, m.Rows*(end-start))
	for i := range m.Rows {
		for j := range end - start {
			data[i*(end-start)+j] = m.Data[i*m.Cols+start+j]
		}
	}
	return NewMatrix(m.Rows, end-start, data)
}

func (m *Matrix) MultiElement(other *Matrix) (result *Matrix) {
	data := make([]float64, m.Rows*m.Cols)
	for i := range data {
		data[i] = m.Data[i] * other.Data[i]
	}
	return NewMatrix(m.Rows, m.Cols, data)
}
