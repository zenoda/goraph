package goraph

import (
	"math"
	"math/rand/v2"
)

/*
Node defines the interface for computing graph nodes.
*/
type Node interface {
	Backward(grad *Matrix)
	Forward() *Matrix
	Reset()
	GetDeps() []Node
}

/*
VariableNode defines a variable node.
*/
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
func (v *VariableNode) GetDeps() []Node {
	return nil
}

/*
AddNode defines a node that performs matrix addition operations.
*/
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
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
}
func (m *AddNode) GetDeps() []Node {
	return []Node{m.X, m.Y}
}

/*
SubNode defines a node that performs matrix subtraction operations.
*/
type SubNode struct {
	X     Node
	Y     Node
	Value *Matrix
}

func Sub(x Node, y Node) *SubNode {
	return &SubNode{
		X: x,
		Y: y,
	}
}
func (m *SubNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.Sub(y)
	}
	return m.Value
}
func (m *SubNode) Backward(grad *Matrix) {
	m.X.Backward(grad)
	m.Y.Backward(grad.Negate())
}
func (m *SubNode) Reset() {
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
}
func (m *SubNode) GetDeps() []Node {
	return []Node{m.X, m.Y}
}

/*
MultiNode defines a node that performs matrix multiplication operations.
*/
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
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
}
func (m *MultiNode) GetDeps() []Node {
	return []Node{m.X, m.Y}
}

/*
MultiElementNode defines a node that performs matrix multiplication based on the
corresponding elements.
*/
type MultiElementNode struct {
	X     Node
	Y     Node
	Value *Matrix
}

func MultiElement(x Node, y Node) *MultiElementNode {
	return &MultiElementNode{
		X: x,
		Y: y,
	}
}
func (m *MultiElementNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.MultiElement(y)
	}
	return m.Value
}
func (m *MultiElementNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	y := m.Y.Forward()
	gradX := NewConstMatrix(x.Rows, x.Cols, 0)
	gradY := NewConstMatrix(y.Rows, y.Cols, 0)
	for i := range grad.Data {
		gradX.Data[i] = y.Data[i] * grad.Data[i]
		gradY.Data[i] = x.Data[i] * grad.Data[i]
	}
	m.X.Backward(gradX)
	m.Y.Backward(gradY)
}
func (m *MultiElementNode) Reset() {
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
}
func (m *MultiElementNode) GetDeps() []Node {
	return []Node{m.X, m.Y}
}

/*
HConcatNode defines a node for matrix horizontal concatenation.
*/
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
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
}
func (m *HConcatNode) GetDeps() []Node {
	return []Node{m.X, m.Y}
}

/*
VConcatNode defines a node for matrix vertical concatenation.
*/
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
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
}
func (m *VConcatNode) GetDeps() []Node {
	return []Node{m.X, m.Y}
}

/*
RowSliceNode defines a node that performs matrix slicing along row direction.
*/
type RowSliceNode struct {
	X          Node
	Start, End int
	Value      *Matrix
}

func RowSlice(x Node, start, end int) *RowSliceNode {
	return &RowSliceNode{
		X:     x,
		Start: start,
		End:   end,
	}
}

func (m *RowSliceNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		m.Value = x.RowSlice(m.Start, m.End)
	}
	return m.Value
}

func (m *RowSliceNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	myGrad := NewConstMatrix(x.Rows, x.Cols, 0)
	for i := range m.End - m.Start {
		for j := range x.Cols {
			myGrad.Data[(i+m.Start)*x.Cols+j] = grad.Data[i*x.Cols+j]
		}
	}
	m.X.Backward(myGrad)
}
func (m *RowSliceNode) Reset() {
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
}
func (m *RowSliceNode) GetDeps() []Node {
	return []Node{m.X}
}

/*
ColSliceNode defines a node that performs matrix slicing along the column direction.
*/
type ColSliceNode struct {
	X          Node
	Start, End int
	Value      *Matrix
}

func ColSlice(x Node, start, end int) *ColSliceNode {
	return &ColSliceNode{
		X:     x,
		Start: start,
		End:   end,
	}
}
func (m *ColSliceNode) Forward() *Matrix {
	if m.Value == nil {
		x := m.X.Forward()
		m.Value = x.ColSlice(m.Start, m.End)
	}
	return m.Value
}
func (m *ColSliceNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	myGrad := NewConstMatrix(x.Rows, x.Cols, 0)
	for i := range x.Rows {
		for j := range m.End - m.Start {
			myGrad.Data[i*x.Cols+j+m.Start] = grad.Data[i*grad.Cols+j]
		}
	}
	m.X.Backward(myGrad)
}
func (m *ColSliceNode) Reset() {
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
}
func (m *ColSliceNode) GetDeps() []Node {
	return []Node{m.X}
}

/*
SigmoidNode defines a node that executes Sigmoid activation function.
*/
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
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
}
func (m *SigmoidNode) GetDeps() []Node {
	return []Node{m.X}
}

/*
ReLuNode defines a node that executes ReLu activation function.
*/
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
				data[i] = 0.001
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
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
}
func (m *ReLuNode) GetDeps() []Node {
	return []Node{m.X}
}

/*
TanhNode defines a node that executes Tanh activation function.
*/
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
		myGrad.Data[i] = (1 - math.Pow(m.Value.Data[i], 2.0) + 0.001) * grad.Data[i]
	}
	m.X.Backward(myGrad)
}
func (m *TanhNode) Reset() {
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
}
func (m *TanhNode) GetDeps() []Node {
	return []Node{m.X}
}

/*
DropoutNode defines a node that performs Dropout operations.
*/
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
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
}
func (m *DropoutNode) GetDeps() []Node {
	return []Node{m.X}
}

/*
SoftmaxNode defines a node that executes the Softmax activation function
*/
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
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
}
func (m *SoftmaxNode) GetDeps() []Node {
	return []Node{m.X}
}

/*
MSELossNode defines a node for calculating mean square error loss.
*/
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
	if grad != nil {
		panic("grad param of loss backward function must be nil")
	}
	x := m.X.Forward()
	y := m.Y.Forward()
	data := make([]float64, x.Rows*x.Cols)
	for i := range data {
		data[i] = (x.Data[i] - y.Data[i]) / float64(x.Cols)
	}
	gx := NewMatrix(x.Rows, x.Cols, data)
	gy := NewConstMatrix(x.Rows, x.Cols, 0.0).Sub(gx)
	m.X.Backward(gx)
	m.Y.Backward(gy)
}

func (m *MSELossNode) Reset() {
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
}
func (m *MSELossNode) GetDeps() []Node {
	return []Node{m.X}
}

/*
CrossEntropyLossNode defines a node dedicated to calculating cross entropy
loss. It should be used in conjunction with the SoftmaxNode, meaning that the
preceding node of this one should be a SoftmaxNode.
*/
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
	if grad != nil {
		panic("grad param of loss backward function must be nil")
	}
	x := m.X.Forward()
	y := m.Y.Forward()
	dataX := make([]float64, x.Rows*x.Cols)
	for i := range dataX {
		if y.Data[i] == 1.0 {
			dataX[i] = -1.0 / x.Data[i]
		} else {
			dataX[i] = 1.0 / (1.0 - x.Data[i])
		}
	}
	gradX := NewMatrix(x.Rows, x.Cols, dataX)
	gradY := NewConstMatrix(y.Rows, y.Cols, 0)
	m.X.Backward(gradX)
	m.Y.Backward(gradY)
}
func (m *CrossEntropyLossNode) Reset() {
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
}
func (m *CrossEntropyLossNode) GetDeps() []Node {
	return []Node{m.X, m.Y}
}

/*
GradThresholdNode defines a processing node that, during forward propagation,
does not perform any processing and directly passes the input to the next step.
In backpropagation, it controls whether to continue propagation based on the set
threshold. When the module of the gradient is less than the threshold,
backpropagation will stop.
*/
type GradThresholdNode struct {
	X         Node
	Value     *Matrix
	threshold float64
}

func GradThreshold(x Node, threshold float64) *GradThresholdNode {
	return &GradThresholdNode{
		X:         x,
		Value:     nil,
		threshold: threshold,
	}
}
func (m *GradThresholdNode) Forward() *Matrix {
	if m.Value == nil {
		m.Value = m.X.Forward()
	}
	return m.Value
}
func (m *GradThresholdNode) Backward(grad *Matrix) {
	mod := 0.0
	for _, v := range grad.Data {
		mod += math.Pow(v, 2)
	}
	mod = math.Sqrt(mod)
	if mod >= m.threshold {
		m.X.Backward(grad)
	}
}
func (m *GradThresholdNode) Reset() {
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
}
func (m *GradThresholdNode) GetDeps() []Node {
	return []Node{m.X}
}
