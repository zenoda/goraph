package goraph

import (
	"fmt"
	"math"
)

type Matrix struct {
	Data []float64
	Rows int
	Cols int
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

func (m *Matrix) Trans() (result *Matrix) {
	data := make([]float64, m.Rows*m.Cols)
	for i := range m.Rows {
		for j := range m.Cols {
			data[j*m.Rows+i] = m.Data[i*m.Cols+j]
		}
	}
	return NewMatrix(m.Cols, m.Rows, data)
}

type Node interface {
	Backward(grad *Matrix)
	Forward() *Matrix
	Reset()
}

// Variable define variable node
type Variable struct {
	Value    *Matrix
	Gradient *Matrix
}

func NewVariable(value *Matrix) *Variable {
	return &Variable{
		Value:    value,
		Gradient: NewConstMatrix(value.Rows, value.Cols, 0.0),
	}
}

func (v *Variable) Forward() *Matrix {
	return v.Value
}
func (v *Variable) Backward(grad *Matrix) {
	v.Gradient = v.Gradient.Add(grad)
}
func (v *Variable) Reset() {
	v.Gradient = NewConstMatrix(v.Value.Rows, v.Value.Cols, 0.0)
}

// Add define add operation node
type Add struct {
	X     Node
	Y     Node
	Value *Matrix
}

func NewAdd(x Node, y Node) *Add {
	return &Add{
		X: x,
		Y: y,
	}
}

func (m *Add) Forward() *Matrix {
	x := m.X.Forward()
	y := m.Y.Forward()
	if m.Value == nil {
		m.Value = x.Add(y)
	}
	return m.Value
}

func (m *Add) Backward(grad *Matrix) {
	m.X.Backward(grad)
	m.Y.Backward(grad)
}

func (m *Add) Reset() {
	m.Value = nil
	m.X.Reset()
	m.Y.Reset()
}

// Multi define multiply operation node
type Multi struct {
	X     Node
	Y     Node
	Value *Matrix
}

func NewMulti(x Node, y Node) *Multi {
	return &Multi{
		X: x,
		Y: y,
	}
}

func (m *Multi) Forward() *Matrix {
	x := m.X.Forward()
	y := m.Y.Forward()
	if m.Value == nil {
		m.Value = x.Multi(y)
	}
	return m.Value
}

func (m *Multi) Backward(grad *Matrix) {
	x := m.X.Forward()
	y := m.Y.Forward()
	m.X.Backward(grad.Multi(y.Trans()))
	m.Y.Backward(x.Trans().Multi(grad))
}

func (m *Multi) Reset() {
	m.Value = nil
	m.X.Reset()
	m.Y.Reset()
}

// Sigmoid define sigmoid function node
type Sigmoid struct {
	X     Node
	Value *Matrix
}

func NewSigmoid(x Node) *Sigmoid {
	return &Sigmoid{
		X: x,
	}
}
func (m *Sigmoid) Forward() *Matrix {
	x := m.X.Forward()
	data := make([]float64, x.Rows*x.Cols)
	for i := range x.Data {
		data[i] = 1.0 / (1.0 + math.Exp(-x.Data[i]))
	}
	if m.Value == nil {
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	return m.Value
}
func (m *Sigmoid) Backward(grad *Matrix) {
	myGrad := NewConstMatrix(m.Value.Rows, m.Value.Cols, 0.0)
	for i := range myGrad.Data {
		myGrad.Data[i] = m.Value.Data[i] * (1 - m.Value.Data[i]) * grad.Data[i]
	}
	m.X.Backward(myGrad)
}
func (m *Sigmoid) Reset() {
	m.Value = nil
	m.X.Reset()
}
