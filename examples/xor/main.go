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
	w1 := NewRandomVariable(2, 5, NewRandFunc(2))
	b1 := NewConstVariable(1, 5, 0.1)
	w2 := NewRandomVariable(5, 1, NewRandFunc(5))
	b2 := NewConstVariable(1, 1, 0.1)

	builder := func() (input, target *VariableNode, output, loss Node) {
		input = NewConstVariable(1, 2, 0)
		target = NewConstVariable(1, 1, 0)
		output = Multi(input, w1)
		output = Add(output, b1)
		output = Tanh(output)
		output = Multi(output, w2)
		output = Add(output, b2)
		output = ReLu(output)
		loss = MSELoss(output, target)
		return
	}
	optimizer := NewSGDOptimizer([]*VariableNode{w1, b1, w2, b2}, 0.2, 0)
	nn := NewNeuralNetwork(builder, optimizer)

	for range 1000 {
		inputData := [][]float64{{0, 0}, {.9, .9}, {0, .9}, {.9, 0}}
		targetData := [][]float64{{0}, {0}, {.9}, {.9}}
		lossValue := nn.Train(inputData, targetData, 4)
		fmt.Printf("Loss: %v\n", lossValue)
	}
}
