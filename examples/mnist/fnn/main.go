package main

import (
	"fmt"
	"goraph"
	"goraph/examples/mnist/dataset"
	"math/rand/v2"
)

func NewRandFunc(num int) func() float64 {
	return func() float64 {
		return (rand.Float64()*2 - 1) / float64(num)
	}
}
func main() {
	w1 := goraph.NewRandomVariable(784, 10, NewRandFunc(784))
	b1 := goraph.NewConstVariable(1, 10, 0.1)
	w2 := goraph.NewRandomVariable(10, 10, NewRandFunc(10))
	b2 := goraph.NewConstVariable(1, 10, 0.1)
	parameters := []*goraph.VariableNode{w1, b1, w2, b2}
	optimizer := goraph.NewSGDOptimizer(parameters, 0.01, 0.9)

	builder := func() (input, target *goraph.VariableNode, output, loss goraph.Node) {
		input = goraph.NewConstVariable(1, 784, 0)
		target = goraph.NewConstVariable(1, 10, 0)

		output = goraph.Multi(input, w1)
		output = goraph.Add(output, b1)
		output = goraph.ReLu(output)

		output = goraph.Multi(output, w2)
		output = goraph.Add(output, b2)
		output = goraph.Softmax(output)

		loss = goraph.CrossEntropyLoss(output, target)
		return
	}

	nn := goraph.NewNeuralNetwork(builder, optimizer)
	model := goraph.NewModel(parameters, nil)
	model.Load("model.json")
	{
		inputData, targetData := dataset.ReadSamples("train")
		for epoch := range 10 {
			lossValue := nn.Train(inputData, targetData, 30)
			fmt.Printf("Epoch: %d, loss: %f\n", epoch, lossValue)
		}
	}
	model.Save("model.json")
	{
		inputData, targetData := dataset.ReadSamples("test")
		lossValue, outputData := nn.Evaluate(inputData, targetData)
		for i := range 10 {
			fmt.Printf("Output: %v, Target: %v\n", outputData[i], targetData[i])
		}
		fmt.Printf("Test, Loss: %v\n", lossValue)
	}
}
