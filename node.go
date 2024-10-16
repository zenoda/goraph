package goraph

import (
	"math"
	"math/rand/v2"
	"sync"
)

/*
Node defines the interface for computing graph nodes.
*/
type Node interface {
	Backward(grad *Matrix)
	Forward() *Matrix
	Reset()
	Tag(name string) Node
}

/*
VariableNode defines a variable node.
*/
type VariableNode struct {
	Name          string  `json:"name"`
	Value         *Matrix `json:"value"`
	Gradient      *Matrix `json:"-"`
	gradientMutex sync.Mutex
}

func NewVariable(rows, cols int, data []float64) *VariableNode {
	return &VariableNode{
		Value:    NewMatrix(rows, cols, data),
		Gradient: NewConstMatrix(rows, cols, 0.0),
	}
}

func NewConstVariable(rows, cols int, value float64) *VariableNode {
	return &VariableNode{
		Value:    NewConstMatrix(rows, cols, value),
		Gradient: NewConstMatrix(rows, cols, 0.0),
	}
}

func NewRandomVariable(rows, cols int, f func() float64) *VariableNode {
	return &VariableNode{
		Value:    NewRandomMatrix(rows, cols, f),
		Gradient: NewConstMatrix(rows, cols, 0.0),
	}
}

func (v *VariableNode) Forward() *Matrix {
	return v.Value
}
func (v *VariableNode) Backward(grad *Matrix) {
	v.gradientMutex.Lock()
	v.Gradient = v.Gradient.Add(grad)
	v.gradientMutex.Unlock()
}
func (v *VariableNode) Reset() {
	v.gradientMutex.Lock()
	v.Gradient = NewConstMatrix(v.Value.Rows, v.Value.Cols, 0.0)
	v.gradientMutex.Unlock()
}
func (v *VariableNode) Tag(name string) Node {
	v.Name = name
	return v
}

/*
AddNode defines a node that performs matrix addition operations.
*/
type AddNode struct {
	X          Node
	Y          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Add(x Node, y Node) *AddNode {
	return &AddNode{
		X: x,
		Y: y,
	}
}

func (m *AddNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.Add(y)
	}
	m.valueMutex.Unlock()
	return m.Value
}

func (m *AddNode) Backward(grad *Matrix) {
	m.X.Backward(grad)
	m.Y.Backward(grad)
}

func (m *AddNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *AddNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
SubNode defines a node that performs matrix subtraction operations.
*/
type SubNode struct {
	X          Node
	Y          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Sub(x Node, y Node) *SubNode {
	return &SubNode{
		X: x,
		Y: y,
	}
}
func (m *SubNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.Sub(y)
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *SubNode) Backward(grad *Matrix) {
	m.X.Backward(grad)
	m.Y.Backward(grad.Negate())
}
func (m *SubNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *SubNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
MultiNode defines a node that performs matrix multiplication operations.
*/
type MultiNode struct {
	X          Node
	Y          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Multi(x Node, y Node) *MultiNode {
	return &MultiNode{
		X: x,
		Y: y,
	}
}

func (m *MultiNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.Multi(y)
	}
	m.valueMutex.Unlock()
	return m.Value
}

func (m *MultiNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	y := m.Y.Forward()
	m.X.Backward(grad.Multi(y.Trans()))
	m.Y.Backward(x.Trans().Multi(grad))
}

func (m *MultiNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *MultiNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
MultiElementNode defines a node that performs matrix multiplication based on the
corresponding elements.
*/
type MultiElementNode struct {
	X          Node
	Y          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func MultiElement(x Node, y Node) *MultiElementNode {
	return &MultiElementNode{
		X: x,
		Y: y,
	}
}
func (m *MultiElementNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.MultiElement(y)
	}
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *MultiElementNode) Tag(name string) Node {
	m.Name = name
	return m
}

type DivElementNode struct {
	X          Node
	Y          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Div(x Node, y Node) *DivElementNode {
	return &DivElementNode{
		X:     x,
		Y:     y,
		Value: nil,
	}
}
func (m *DivElementNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.DivElement(y)
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *DivElementNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	y := m.Y.Forward()
	gradX := grad.DivElement(y)
	gradY := grad.MultiElement(x)
	m.X.Backward(gradX)
	m.Y.Backward(gradY)
}
func (m *DivElementNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *DivElementNode) Tag(name string) Node {
	m.Name = name
	return m
}

type LogNode struct {
	X          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Log(x Node) *LogNode {
	return &LogNode{
		X:     x,
		Value: nil,
	}
}
func (m *LogNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i := range data {
			data[i] = math.Log(x.Data[i])
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *LogNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	gradX := NewConstMatrix(x.Rows, x.Cols, 0)
	for i := range x.Data {
		gradX.Data[i] = grad.Data[i] / x.Data[i]
	}
	m.X.Backward(gradX)
}
func (m *LogNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *LogNode) Tag(name string) Node {
	m.Name = name
	return m
}

type TransNode struct {
	X          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Trans(x Node) *TransNode {
	return &TransNode{
		X:     x,
		Value: nil,
	}
}
func (m *TransNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		m.Value = m.X.Forward().Trans()
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *TransNode) Backward(grad *Matrix) {
	gradX := grad.Trans()
	m.X.Backward(gradX)
}
func (m *TransNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *TransNode) Tag(name string) Node {
	m.Name = name
	return m
}

type ReshapeNode struct {
	X          Node
	Rows       int
	Cols       int
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Reshape(x Node, rows, cols int) *ReshapeNode {
	return &ReshapeNode{
		X:     x,
		Rows:  rows,
		Cols:  cols,
		Value: nil,
	}
}
func (m *ReshapeNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		m.Value = x.Reshape(m.Rows, m.Cols)
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *ReshapeNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	xGrad := NewConstMatrix(x.Rows, x.Cols, 0)
	copy(xGrad.Data, grad.Data)
	m.X.Backward(xGrad)
}
func (m *ReshapeNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *ReshapeNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
HConcatNode defines a node for matrix horizontal concatenation.
*/
type HConcatNode struct {
	X          Node
	Y          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func HConcat(x Node, y Node) *HConcatNode {
	return &HConcatNode{
		X: x,
		Y: y,
	}
}

func (m *HConcatNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.HConcat(y)
	}
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *HConcatNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
VConcatNode defines a node for matrix vertical concatenation.
*/
type VConcatNode struct {
	X          Node
	Y          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func VConcat(x Node, y Node) *VConcatNode {
	return &VConcatNode{
		X: x,
		Y: y,
	}
}
func (m *VConcatNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		y := m.Y.Forward()
		m.Value = x.VConcat(y)
	}
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *VConcatNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
RowSliceNode defines a node that performs matrix slicing along row direction.
*/
type RowSliceNode struct {
	X          Node
	Start, End int
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func RowSlice(x Node, start, end int) *RowSliceNode {
	return &RowSliceNode{
		X:     x,
		Start: start,
		End:   end,
	}
}

func (m *RowSliceNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		m.Value = x.RowSlice(m.Start, m.End)
	}
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *RowSliceNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
ColSliceNode defines a node that performs matrix slicing along the column direction.
*/
type ColSliceNode struct {
	X          Node
	Start, End int
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func ColSlice(x Node, start, end int) *ColSliceNode {
	return &ColSliceNode{
		X:     x,
		Start: start,
		End:   end,
	}
}
func (m *ColSliceNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		m.Value = x.ColSlice(m.Start, m.End)
	}
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *ColSliceNode) Tag(name string) Node {
	m.Name = name
	return m
}

type RowSumNode struct {
	X          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func RowSum(x Node) *RowSumNode {
	return &RowSumNode{
		X:     x,
		Value: nil,
	}
}
func (m *RowSumNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		m.Value = x.RowSum()
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *RowSumNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	dataX := make([]float64, x.Rows*x.Cols)
	for i := range x.Rows {
		for j := range x.Cols {
			dataX[i*x.Cols+j] = grad.Data[i]
		}
	}
	gradX := NewMatrix(x.Rows, x.Cols, dataX)
	m.X.Backward(gradX)
}
func (m *RowSumNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *RowSumNode) Tag(name string) Node {
	m.Name = name
	return m
}

type ColSumNode struct {
	X          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func ColSum(x Node) *ColSumNode {
	return &ColSumNode{
		X:     x,
		Value: nil,
	}
}
func (m *ColSumNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		m.Value = x.ColSum()
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *ColSumNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	dataX := make([]float64, x.Rows*x.Cols)
	for i := range x.Rows {
		for j := range x.Cols {
			dataX[i*x.Cols+j] = grad.Data[j]
		}
	}
	gradX := NewMatrix(x.Rows, x.Cols, dataX)
	m.X.Backward(gradX)
}
func (m *ColSumNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *ColSumNode) Tag(name string) Node {
	m.Name = name
	return m
}

type ScaleNode struct {
	X          Node
	Rate       float64
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Scale(x Node, rate float64) *ScaleNode {
	return &ScaleNode{
		X:     x,
		Rate:  rate,
		Value: nil,
	}
}
func (m *ScaleNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		m.Value = x.Scale(m.Rate)
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *ScaleNode) Backward(grad *Matrix) {
	gradX := grad.Scale(m.Rate)
	m.X.Backward(gradX)
}
func (m *ScaleNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *ScaleNode) Tag(name string) Node {
	m.Name = name
	return m
}

type ValueThresholdNode struct {
	X          Node
	Value      *Matrix
	MinValue   float64
	MaxValue   float64
	Name       string
	valueMutex sync.Mutex
}

func ValueThreshold(x Node, minVal, maxVal float64) *ValueThresholdNode {
	return &ValueThresholdNode{
		X:        x,
		Value:    nil,
		MinValue: minVal,
		MaxValue: maxVal,
	}
}
func (m *ValueThresholdNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i := range x.Data {
			data[i] = min(m.MaxValue, max(m.MinValue, x.Data[i]))
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *ValueThresholdNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	gradX := NewConstMatrix(x.Rows, x.Cols, 0)
	for i := range grad.Data {
		if x.Data[i] == m.MinValue || x.Data[i] == m.MaxValue {
			gradX.Data[i] = 0
		} else {
			gradX.Data[i] = grad.Data[i]
		}
	}
	m.X.Backward(gradX)
}
func (m *ValueThresholdNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *ValueThresholdNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
SigmoidNode defines a node that executes Sigmoid activation function.
*/
type SigmoidNode struct {
	X          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Sigmoid(x Node) *SigmoidNode {
	return &SigmoidNode{
		X: x,
	}
}
func (m *SigmoidNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i := range x.Data {
			data[i] = 1.0 / (1.0 + math.Exp(-x.Data[i]))
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *SigmoidNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
ReLuNode defines a node that executes ReLu activation function.
*/
type ReLuNode struct {
	X          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func ReLu(x Node) *ReLuNode {
	return &ReLuNode{
		X: x,
	}
}

func (m *ReLuNode) Forward() *Matrix {
	m.valueMutex.Lock()
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
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *ReLuNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
TanhNode defines a node that executes Tanh activation function.
*/
type TanhNode struct {
	X          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Tanh(x Node) *TanhNode {
	return &TanhNode{
		X: x,
	}
}
func (m *TanhNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i := range data {
			data[i] = (math.Exp(x.Data[i]) - math.Exp(-x.Data[i])) / (math.Exp(x.Data[i]) + math.Exp(-x.Data[i]))
			if math.IsNaN(data[i]) {
				panic("The item is NaN.")
			}
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *TanhNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
DropoutNode defines a node that performs Dropout operations.
*/
type DropoutNode struct {
	X          Node
	P          float64 //Keep probability
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Dropout(x Node, p float64) *DropoutNode {
	return &DropoutNode{
		X: x,
		P: p,
	}
}
func (m *DropoutNode) Forward() *Matrix {
	m.valueMutex.Lock()
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
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *DropoutNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
SoftmaxNode defines a node that executes the Softmax activation function
*/
type SoftmaxNode struct {
	X          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func Softmax(x Node) *SoftmaxNode {
	return &SoftmaxNode{
		X: x,
	}
}
func (m *SoftmaxNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		data := make([]float64, x.Rows*x.Cols)
		for i := range x.Rows {
			sum := 0.0
			values := make([]float64, x.Cols)
			for j := range x.Cols {
				values[j] = math.Exp(x.Data[i*x.Cols+j])
				if math.IsInf(values[j], 0) {
					panic("The value is infinity.")
				}
				if math.IsNaN(values[j]) {
					panic("The value is NaN.")
				}
				sum += values[j]
			}
			if sum == 0.0 {
				panic("Value of sum must not be 0.")
			}
			for j, v := range values {
				data[i*x.Cols+j] = v / sum
			}
		}
		m.Value = NewMatrix(x.Rows, x.Cols, data)
	}
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *SoftmaxNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
MSELossNode defines a node for calculating mean square error loss.
*/
type MSELossNode struct {
	X          Node
	Y          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func MSELoss(x Node, y Node) *MSELossNode {
	return &MSELossNode{
		X: x,
		Y: y,
	}
}

func (m *MSELossNode) Forward() *Matrix {
	m.valueMutex.Lock()
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
	m.valueMutex.Unlock()
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
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *MSELossNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
CrossEntropyLossNode defines a node dedicated to calculating cross entropy
loss. It should be used in conjunction with the SoftmaxNode, meaning that the
preceding node of this one should be a SoftmaxNode.
*/
type CrossEntropyLossNode struct {
	X          Node
	Y          Node
	Value      *Matrix
	Name       string
	valueMutex sync.Mutex
}

func CrossEntropyLoss(x Node, y Node) *CrossEntropyLossNode {
	return &CrossEntropyLossNode{
		X: x,
		Y: y,
	}
}
func (m *CrossEntropyLossNode) Forward() *Matrix {
	m.valueMutex.Lock()
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
	m.valueMutex.Unlock()
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
			if x.Data[i] == 0.0 {
				dataX[i] = -1.0 / 0.001
			} else {
				dataX[i] = -1.0 / x.Data[i]
			}
		} else {
			if 1.0-x.Data[i] == 0.0 {
				dataX[i] = 1.0 / 0.001
			} else {
				dataX[i] = 1.0 / (1.0 - x.Data[i])
			}
		}
	}
	gradX := NewMatrix(x.Rows, x.Cols, dataX)
	gradY := NewConstMatrix(y.Rows, y.Cols, 0)
	m.X.Backward(gradX)
	m.Y.Backward(gradY)
}
func (m *CrossEntropyLossNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
		m.Y.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *CrossEntropyLossNode) Tag(name string) Node {
	m.Name = name
	return m
}

/*
GradThresholdNode defines a processing node that, during forward propagation,
does not perform any processing and directly passes the input to the next step.
In backpropagation, it controls whether to continue propagation based on the set
Threshold. When the module of the gradient is less than the Threshold,
backpropagation will stop.
*/
type GradThresholdNode struct {
	X          Node
	Value      *Matrix
	Threshold  float64
	Name       string
	valueMutex sync.Mutex
}

func GradThreshold(x Node, threshold float64) *GradThresholdNode {
	return &GradThresholdNode{
		X:         x,
		Value:     nil,
		Threshold: threshold,
	}
}
func (m *GradThresholdNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		m.Value = m.X.Forward()
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *GradThresholdNode) Backward(grad *Matrix) {
	mod := 0.0
	for _, v := range grad.Data {
		mod += math.Pow(v, 2)
	}
	mod = math.Sqrt(mod)
	if mod >= m.Threshold {
		m.X.Backward(grad)
	}
}
func (m *GradThresholdNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *GradThresholdNode) Tag(name string) Node {
	m.Name = name
	return m
}

type PoolNode struct {
	X          Node
	Width      int
	Height     int
	Stride     int
	Value      *Matrix
	Flags      []int
	Name       string
	valueMutex sync.Mutex
}

func Pool(x Node, width, height, stride int) *PoolNode {
	return &PoolNode{
		X:      x,
		Width:  width,
		Height: height,
		Stride: stride,
		Value:  nil,
		Flags:  nil,
	}
}
func (m *PoolNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		var xPadding, xSteps, yPadding, ySteps int
		if (x.Cols-m.Width)%m.Stride == 0 {
			xPadding = 0
			xSteps = (x.Cols-m.Width)/m.Stride + 1
		} else {
			xPadding = m.Stride - (x.Cols-m.Width)%m.Stride
			xSteps = (x.Cols-m.Width+xPadding)/m.Stride + 1
		}
		if (x.Rows-m.Height)%m.Stride == 0 {
			yPadding = 0
			ySteps = (x.Rows-m.Height)/m.Stride + 1
		} else {
			yPadding = m.Stride - (x.Rows-m.Height)%m.Stride
			ySteps = (x.Rows-m.Height+yPadding)/m.Stride + 1
		}
		data := make([]float64, xSteps*ySteps)
		m.Flags = make([]int, xSteps*ySteps)
		for i := range xSteps {
			for j := range ySteps {
				maxVal := math.Inf(-1)
				maxValIdx := 0
				for w := range m.Width {
					colIdx := i*m.Stride + w - xPadding/2
					if colIdx < 0 || colIdx >= x.Cols {
						maxVal = max(maxVal, 0)
						continue
					}
					for h := range m.Height {
						rowIdx := j*m.Stride + h - yPadding/2
						if rowIdx < 0 || rowIdx >= x.Rows {
							maxVal = max(maxVal, 0)
							continue
						}
						if x.Data[rowIdx*x.Cols+colIdx] > maxVal {
							maxVal = x.Data[rowIdx*x.Cols+colIdx]
							maxValIdx = rowIdx*x.Cols + colIdx
						}
					}
				}
				data[j*xSteps+i] = maxVal
				m.Flags[j*xSteps+i] = maxValIdx
			}
		}
		m.Value = NewMatrix(ySteps, xSteps, data)
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *PoolNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	xGrad := NewConstMatrix(x.Rows, x.Cols, 0)
	for i, idx := range m.Flags {
		xGrad.Data[idx] = grad.Data[i]
	}
	m.X.Backward(xGrad)
}
func (m *PoolNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.Flags = nil
		m.X.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *PoolNode) Tag(name string) Node {
	m.Name = name
	return m
}

type ConvNode struct {
	X          Node
	Kernel     Node
	Stride     int
	Value      *Matrix
	XPadding   int
	YPadding   int
	Name       string
	valueMutex sync.Mutex
}

func Conv(x Node, kernel Node, stride int) *ConvNode {
	return &ConvNode{
		X:        x,
		Kernel:   kernel,
		Stride:   stride,
		Value:    nil,
		XPadding: 0,
		YPadding: 0,
	}
}
func (m *ConvNode) Forward() *Matrix {
	m.valueMutex.Lock()
	if m.Value == nil {
		x := m.X.Forward()
		kernel := m.Kernel.Forward()
		var xPadding, xSteps, yPadding, ySteps int
		if x.Cols%m.Stride == 0 {
			xPadding = kernel.Cols
			xSteps = x.Cols / m.Stride
		} else {
			xPadding = kernel.Cols + (m.Stride - x.Cols%m.Stride)
			xSteps = (x.Cols + m.Stride - x.Cols%m.Stride) / m.Stride
		}
		if x.Rows%m.Stride == 0 {
			yPadding = kernel.Rows
			ySteps = x.Rows / m.Stride
		} else {
			yPadding = kernel.Rows + (m.Stride - x.Rows%m.Stride)
			ySteps = (x.Rows + m.Stride - x.Rows%m.Stride) / m.Stride
		}
		data := make([]float64, xSteps*ySteps)
		for i := range xSteps {
			for j := range ySteps {
				sumVal := 0.0
				for kx := range kernel.Cols {
					colIdx := i*m.Stride + kx - xPadding/2
					if colIdx < 0 || colIdx >= x.Cols {
						continue
					}
					for ky := range kernel.Rows {
						rowIdx := j*m.Stride + ky - yPadding/2
						if rowIdx < 0 || rowIdx >= x.Rows {
							continue
						}
						sumVal += x.Data[rowIdx*x.Cols+colIdx] * kernel.Data[ky*kernel.Cols+kx]
					}
				}
				data[j*xSteps+i] = sumVal
			}
		}
		m.Value = NewMatrix(ySteps, xSteps, data)
		m.XPadding = xPadding
		m.YPadding = yPadding
	}
	m.valueMutex.Unlock()
	return m.Value
}
func (m *ConvNode) Backward(grad *Matrix) {
	x := m.X.Forward()
	kernel := m.Kernel.Forward()
	xGrad := NewConstMatrix(x.Rows, x.Cols, 0)
	kernelGrad := NewConstMatrix(kernel.Rows, kernel.Cols, 0)

	for gr := range grad.Rows {
		for gc := range grad.Cols {
			for kr := range kernel.Rows {
				rowIdx := gr*m.Stride + kr - m.YPadding/2
				if rowIdx < 0 || rowIdx >= x.Rows {
					continue
				}
				for kc := range kernel.Cols {
					colIdx := gc*m.Stride + kc - m.XPadding/2
					if colIdx < 0 || colIdx >= x.Cols {
						continue
					}
					xGrad.Data[rowIdx*x.Cols+colIdx] += kernel.Data[kr*kernel.Cols+kc] * grad.Data[gr*grad.Cols+gc]
				}
			}
		}
	}

	for i := range kernelGrad.Rows {
		for j := range kernelGrad.Cols {
			for gr := range grad.Rows {
				rowIdx := gr*m.Stride + i - m.YPadding/2
				if rowIdx < 0 || rowIdx >= x.Rows {
					continue
				}
				for gc := range grad.Cols {
					colIdx := gc*m.Stride + j - m.XPadding/2
					if colIdx < 0 || colIdx >= x.Cols {
						continue
					}
					kernelGrad.Data[i*kernelGrad.Cols+j] += x.Data[rowIdx*x.Cols+colIdx] * grad.Data[gr*grad.Cols+gc]
				}
			}
		}
	}
	m.X.Backward(xGrad)
	m.Kernel.Backward(kernelGrad)
}
func (m *ConvNode) Reset() {
	m.valueMutex.Lock()
	if m.Value != nil {
		m.Value = nil
		m.XPadding = 0
		m.YPadding = 0
		m.X.Reset()
		m.Kernel.Reset()
	}
	m.valueMutex.Unlock()
}
func (m *ConvNode) Tag(name string) Node {
	m.Name = name
	return m
}
