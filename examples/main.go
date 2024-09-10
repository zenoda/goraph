package main

import (
	"fmt"
	. "goraph"
	"math/rand/v2"
)

func main() {
	input := NewVariable(4, 2, []float64{0, 0, .9, .9, 0, .9, .9, 0})
	w1 := NewRandomVariable(2, 5, rand.Float64)
	b1 := NewConstVariable(1, 5, 0.1)
	w2 := NewRandomVariable(5, 1, rand.Float64)
	b2 := NewConstVariable(1, 1, 0.1)
	target := NewVariable(4, 1, []float64{0, 0, .9, .9})

	var output Node
	output = Multi(input, w1)
	output = Add(output, Multi(NewConstVariable(4, 1, 1), b1))
	output = Tanh(output)

	output = Multi(output, w2)
	output = Add(output, Multi(NewConstVariable(4, 1, 1), b2))
	output = Tanh(output)

	var loss Node
	loss = MSELoss(output, target)
	optimizer := NewSGDOptimizer([]*VariableNode{w1, b1, w2, b2}, 0.1, 0.99)
	for range 600 {
		outputValue := output.Forward()
		fmt.Println(outputValue)
		lossValue := loss.Forward()
		fmt.Println(lossValue)
		loss.Backward(nil)
		optimizer.Step()
		loss.Reset()
	}
}
