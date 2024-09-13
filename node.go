package goraph

import (
	"encoding/json"
	"io"
	"math"
	"math/rand/v2"
)

// Node define the graph node interface
type Node interface {
	Backward(grad *Matrix)
	Forward() *Matrix
	Reset()
}

// VariableNode define variable node
type VariableNode struct {
	Name     string  `json:"name"`
	Value    *Matrix `json:"value"`
	Gradient *Matrix `json:"-"`
}

func NewVariable(rows, cols int, data []float64, name string) *VariableNode {
	return &VariableNode{
		Name:     name,
		Value:    NewMatrix(rows, cols, data),
		Gradient: NewConstMatrix(rows, cols, 0.0),
	}
}

func NewConstVariable(rows, cols int, value float64, name string) *VariableNode {
	return &VariableNode{
		Name:     name,
		Value:    NewConstMatrix(rows, cols, value),
		Gradient: NewConstMatrix(rows, cols, 0.0),
	}
}

func NewRandomVariable(rows, cols int, f func() float64, name string) *VariableNode {
	return &VariableNode{
		Name:     name,
		Value:    NewRandomMatrix(rows, cols, f),
		Gradient: NewConstMatrix(rows, cols, 0.0),
	}
}

func (v *VariableNode) Forward() *Matrix {
	return v.Value
}
func (v *VariableNode) Backward(grad *Matrix) {
	v.Gradient = v.Gradient.Add(grad)
}
func (v *VariableNode) Reset() {
	v.Gradient = NewConstMatrix(v.Value.Rows, v.Value.Cols, 0.0)
}

//--------------------------
// Operational functions
//--------------------------

// AddNode define add operation node
type AddNode struct {
	X     Node
	Y     Node
	Value *Matrix
}

func Add(x Node, y Node) *AddNode {
	return &AddNode{
		X: x,
		Y: y,
	}
}

func (m *AddNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.Add(y)
	}
	return m.Value
}

func (m *AddNode) Backward(grad *Matrix) {
	m.X.Backward(grad)
	m.Y.Backward(grad)
}

func (m *AddNode) Reset() {
	m.Value = nil
	m.X.Reset()
	m.Y.Reset()
}

// MultiNode define multiply operation node
type MultiNode struct {
	X     Node
	Y     Node
	Value *Matrix
}

func Multi(x Node, y Node) *MultiNode {
	return &MultiNode{
		X: x,
		Y: y,
	}
}

func (m *MultiNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.Multi(y)
	}
	return m.Value
}

func (m *MultiNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	y := m.Y.Forward()
	m.X.Backward(grad.Multi(y.Trans()))
	m.Y.Backward(x.Trans().Multi(grad))
}

func (m *MultiNode) Reset() {
	m.Value = nil
	m.X.Reset()
	m.Y.Reset()
}

// HConcatNode define matrix horizontal concatenation function
type HConcatNode struct {
	X     Node
	Y     Node
	Value *Matrix
}

func HConcat(x Node, y Node) *HConcatNode {
	return &HConcatNode{
		X: x,
		Y: y,
	}
}

func (m *HConcatNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.HConcat(y)
	}
	return m.Value
}

func (m *HConcatNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	y := m.Y.Forward()
	var dataX, dataY []float64
	for i := range x.Rows {
		dataX = append(dataX, grad.Data[i*grad.Cols:i*grad.Cols+x.Cols]...)
		dataY = append(dataY, grad.Data[i*grad.Cols+x.Cols:i*grad.Cols+grad.Cols]...)
	}
	gradX := NewMatrix(x.Rows, x.Cols, dataX)
	gradY := NewMatrix(y.Rows, y.Cols, dataY)
	m.X.Backward(gradX)
	m.Y.Backward(gradY)
}
func (m *HConcatNode) Reset() {
	m.Value = nil
	m.X.Reset()
	m.Y.Reset()
}

// VConcatNode define matrix vertical concatenation function
type VConcatNode struct {
	X     Node
	Y     Node
	Value *Matrix
}

func VConcat(x Node, y Node) *VConcatNode {
	return &VConcatNode{
		X: x,
		Y: y,
	}
}
func (m *VConcatNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.VConcat(y)
	}
	return m.Value
}
func (m *VConcatNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	y := m.Y.Forward()
	dataX := grad.Data[:x.Rows*grad.Cols]
	dataY := grad.Data[x.Rows*grad.Cols:]
	gradX := NewMatrix(x.Rows, x.Cols, dataX)
	gradY := NewMatrix(y.Rows, y.Cols, dataY)
	m.X.Backward(gradX)
	m.Y.Backward(gradY)
}
func (m *VConcatNode) Reset() {
	m.Value = nil
	m.X.Reset()
	m.Y.Reset()
}

// -----------------
// Activation functions
// -------------------

// SigmoidNode define sigmoid function node
type SigmoidNode struct {
	X     Node
	Value *Matrix
}

func Sigmoid(x Node) *SigmoidNode {
	return &SigmoidNode{
		X: x,
	}
}
func (m *SigmoidNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i := range x.Data {
			data[i] = 1.0 / (1.0 + math.Exp(-x.Data[i]))
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	return m.Value
}
func (m *SigmoidNode) Backward(grad *Matrix) {
	myGrad := NewConstMatrix(m.Value.Rows, m.Value.Cols, 0.0)
	for i := range myGrad.Data {
		myGrad.Data[i] = m.Value.Data[i] * (1 - m.Value.Data[i]) * grad.Data[i]
	}
	m.X.Backward(myGrad)
}
func (m *SigmoidNode) Reset() {
	m.Value = nil
	m.X.Reset()
}

// ReLuNode define ReLu activation function
type ReLuNode struct {
	X     Node
	Value *Matrix
}

func ReLu(x Node) *ReLuNode {
	return &ReLuNode{
		X: x,
	}
}

func (m *ReLuNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i, v := range x.Data {
			if v > 0 {
				data[i] = v
			} else {
				data[i] = 0.01
			}
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	return m.Value
}
func (m *ReLuNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	myGrad := NewConstMatrix(m.Value.Rows, m.Value.Cols, 0.0)
	for i, v := range x.Data {
		if v > 0 {
			myGrad.Data[i] = grad.Data[i]
		} else {
			myGrad.Data[i] = grad.Data[i] * 0.01
		}
	}
	m.X.Backward(myGrad)
}
func (m *ReLuNode) Reset() {
	m.Value = nil
	m.X.Reset()
}

// TanhNode define Tanh activation function
type TanhNode struct {
	X     Node
	Value *Matrix
}

func Tanh(x Node) *TanhNode {
	return &TanhNode{
		X: x,
	}
}
func (m *TanhNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i := range data {
			data[i] = (math.Exp(x.Data[i]) - math.Exp(-x.Data[i])) / (math.Exp(x.Data[i]) + math.Exp(-x.Data[i]))
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	return m.Value
}

func (m *TanhNode) Backward(grad *Matrix) {
	myGrad := NewConstMatrix(m.Value.Rows, m.Value.Cols, 0.0)
	for i := range myGrad.Data {
		myGrad.Data[i] = (1 - math.Pow(m.Value.Data[i], 2.0) + 0.01) * grad.Data[i]
	}
	m.X.Backward(myGrad)
}
func (m *TanhNode) Reset() {
	m.Value = nil
	m.X.Reset()
}

// DropoutNode define Dropout function
type DropoutNode struct {
	X     Node
	P     float64 //Keep probability
	Value *Matrix
}

func Dropout(x Node, p float64) *DropoutNode {
	return &DropoutNode{
		X: x,
		P: p,
	}
}
func (m *DropoutNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i := range data {
			if rand.Float64() < m.P {
				data[i] = x.Data[i]
			} else {
				data[i] = 0
			}
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	return m.Value
}
func (m *DropoutNode) Backward(grad *Matrix) {
	myGrad := NewConstMatrix(m.Value.Rows, m.Value.Cols, 0.0)
	for i := range myGrad.Data {
		if m.Value.Data[i] == 0 {
			myGrad.Data[i] = 0
		} else {
			myGrad.Data[i] = grad.Data[i]
		}
	}
	m.X.Backward(myGrad)
}
func (m *DropoutNode) Reset() {
	m.Value = nil
	m.X.Reset()
}

// SoftmaxNode define softmax activation function
type SoftmaxNode struct {
	X     Node
	Value *Matrix
}

func Softmax(x Node) *SoftmaxNode {
	return &SoftmaxNode{
		X: x,
	}
}
func (m *SoftmaxNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i := range x.Rows {
			sum := 0.0
			values := make([]float64, x.Cols)
			for j := range x.Cols {
				values[j] = math.Exp(x.Data[i*x.Cols+j])
				sum += values[j]
			}
			for j, v := range values {
				data[i*x.Cols+j] = v / sum
			}
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	return m.Value
}
func (m *SoftmaxNode) Backward(grad *Matrix) {
	myGrad := NewConstMatrix(m.Value.Rows, m.Value.Cols, 0.0)
	for i := range myGrad.Data {
		myGrad.Data[i] = m.Value.Data[i] * (1 - m.Value.Data[i]) * grad.Data[i]
	}
	m.X.Backward(myGrad)
}
func (m *SoftmaxNode) Reset() {
	m.Value = nil
	m.X.Reset()
}

//------------------------
// Loss functions
//------------------------

type MSELossNode struct {
	X     Node
	Y     Node
	Value *Matrix
}

func MSELoss(x Node, y Node) *MSELossNode {
	return &MSELossNode{
		X: x,
		Y: y,
	}
}

func (m *MSELossNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		if x.Rows != y.Rows || x.Cols != y.Cols {
			panic("Matrix dimensions do not match")
		}
		data := make([]float64, 1)
		for i := range x.Rows {
			loss := 0.0
			for j := range x.Cols {
				loss += math.Pow(x.Data[i*x.Cols+j]-y.Data[i*x.Cols+j], 2)
			}
			data[0] += loss / float64(x.Cols)
		}
		data[0] /= float64(x.Rows)
		m.Value = NewMatrix(1, 1, data)
	}
	return m.Value
}

func (m *MSELossNode) Backward(grad *Matrix) {
	if grad == nil {
		grad = NewConstMatrix(1, 1, 1)
	}
	x := m.X.Forward()
	y := m.Y.Forward()
	data := make([]float64, x.Rows*x.Cols)
	for i := range data {
		data[i] = grad.Data[0] / float64(x.Rows) / float64(x.Cols) * 2 * (x.Data[i] - y.Data[i])
	}
	gx := NewMatrix(x.Rows, x.Cols, data)
	gy := NewConstMatrix(x.Rows, x.Cols, 0.0).Sub(gx)
	m.X.Backward(gx)
	m.Y.Backward(gy)
}

func (m *MSELossNode) Reset() {
	m.Value = nil
	m.X.Reset()
	m.Y.Reset()
}

// CrossEntropyLossNode define cross entropy loss function
type CrossEntropyLossNode struct {
	X     Node
	Y     Node
	Value *Matrix
}

func CrossEntropyLoss(x Node, y Node) *CrossEntropyLossNode {
	return &CrossEntropyLossNode{
		X: x,
		Y: y,
	}
}
func (m *CrossEntropyLossNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		data := make([]float64, 1)
		for i, vy := range y.Data {
			if vy == 1.0 {
				data[0] += -math.Log(x.Data[i])
			}
		}
		data[0] /= float64(x.Rows)
		m.Value = NewMatrix(1, 1, data)
	}
	return m.Value
}
func (m *CrossEntropyLossNode) Backward(grad *Matrix) {
	if grad == nil {
		grad = NewConstMatrix(1, 1, 1)
	}
	x := m.X.Forward()
	y := m.Y.Forward()
	dataX := make([]float64, x.Rows*x.Cols)
	for i := range dataX {
		if y.Data[i] == 1.0 {
			dataX[i] = -1.0 / x.Data[i] * grad.Data[0] / float64(x.Rows)
		} else {
			dataX[i] = 1.0 / (1.0 - x.Data[i]) * grad.Data[0] / float64(x.Rows)
		}
	}
	gradX := NewMatrix(x.Rows, x.Cols, dataX)
	gradY := NewConstMatrix(y.Rows, y.Cols, 0)
	m.X.Backward(gradX)
	m.Y.Backward(gradY)
}
func (m *CrossEntropyLossNode) Reset() {
	m.Value = nil
	m.X.Reset()
	m.Y.Reset()
}

//-----------------------------
//Storage
//-----------------------------

func Save(writer io.Writer, params ...*VariableNode) error {
	jsonData, err := json.Marshal(params)
	if err != nil {
		return err
	}
	_, err = writer.Write(jsonData)
	return err
}

func Load(reader io.Reader, params ...*VariableNode) error {
	jsonData, err := io.ReadAll(reader)
	if err != nil {
		return err
	}
	var paramsData []*VariableNode
	err = json.Unmarshal(jsonData, &paramsData)
	if err != nil {
		return err
	}
	for _, param := range params {
		for _, pd := range paramsData {
			if param.Name == pd.Name {
				param.Value = pd.Value
				break
			}
		}
	}
	return nil
}
