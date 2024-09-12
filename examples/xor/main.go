package main

import (
	"fmt"
	. "goraph"
	"math/rand/v2"
)

func NewRandFunc(num int) func() float64 {
	return func() float64 {
		return (rand.Float64()*2 - 1) / float64(num)
	}
}
func main() {
	input := NewVariable(4, 2, []float64{0, 0, .9, .9, 0, .9, .9, 0}, "input")
	w1 := NewRandomVariable(2, 5, NewRandFunc(2), "w1")
	b1 := NewConstVariable(1, 5, 0.1, "b1")
	w2 := NewRandomVariable(5, 1, NewRandFunc(5), "w2")
	b2 := NewConstVariable(1, 1, 0.1, "b2")
	target := NewVariable(4, 1, []float64{0, 0, .9, .9}, "target")

	var output Node
	output = Multi(input, w1)
	output = Add(output, Multi(NewConstVariable(4, 1, 1, "bb1"), b1))
	output = Tanh(output)

	output = Multi(output, w2)
	output = Add(output, Multi(NewConstVariable(4, 1, 1, "bb2"), b2))
	output = ReLu(output)

	var loss Node
	loss = MSELoss(output, target)
	optimizer := NewSGDOptimizer([]*VariableNode{w1, b1, w2, b2}, 0.1, 0)
	for range 2000 {
		outputValue := output.Forward()
		lossValue := loss.Forward()
		fmt.Printf("Loss: %v, Output: %v\n", lossValue, outputValue)
		loss.Backward(nil)
		optimizer.Step()
		loss.Reset()
	}
}
