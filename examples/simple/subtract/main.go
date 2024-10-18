package main

import (
	"fmt"
	. "github.com/zenoda/goraph"
	"math"
	"math/rand/v2"
)

func NewRandFunc(num int) func() float64 {
	return func() float64 {
		return (rand.Float64()*2 - 1) / math.Sqrt(float64(num))
	}
}
func main() {
	w1 := NewRandomVariable(2, 8, NewRandFunc(2))
	b1 := NewConstVariable(1, 8, 0.1)
	w2 := NewRandomVariable(8, 1, NewRandFunc(200))
	b2 := NewConstVariable(1, 1, 0.1)

	builder := func() (input, target *VariableNode, output, loss Node) {
		input = NewConstVariable(1, 2, 0)
		target = NewConstVariable(1, 1, 0)
		output = Multi(input, w1)
		output = Multi(output, w2)
		loss = MSELoss(output, target)
		return
	}
	parameters := []*VariableNode{w1, b1, w2, b2}
	optimizer := NewSGDOptimizer(parameters, 0.2, 0)
	nn := NewNeuralNetwork(builder, optimizer)
	{
		var inputData [][]float64
		var targetData [][]float64
		for i := range 10 {
			for j := range 10 {
				inputData = append(inputData, []float64{float64(i) / 10, float64(j) / 10})
				targetData = append(targetData, []float64{float64(i)/10 - float64(j)/10})
			}
		}

		for range 1000 {
			lossValue := nn.Train(inputData, targetData, 4)
			fmt.Printf("Training: Loss: %v\n", lossValue)
		}
	}
	{
		inputData := [][]float64{{0, 0.85}, {0.11, 0.9}, {0.6, 0.7}, {0.8, 0.7}}
		targetData := [][]float64{{-0.85}, {-0.79}, {-0.1}, {0.1}}
		lossValue, output := nn.Evaluate(inputData, targetData)
		for i, v := range output {
			fmt.Println(i, v, targetData[i])
		}
		fmt.Printf("Evaluating: Loss: %v\n", lossValue)
	}
}
