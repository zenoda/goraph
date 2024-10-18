package main

import (
	"fmt"
	"github.com/zenoda/goraph"
)

var (
	pw = goraph.NewVariable(1, 2, []float64{0.1, 0.5})
	w1 = goraph.NewRandomVariable(2, 32, goraph.NewKaimingNormalInit(2))
	b1 = goraph.NewConstVariable(1, 32, 0)
	w2 = goraph.NewRandomVariable(32, 32, goraph.NewKaimingNormalInit(32))
	b2 = goraph.NewConstVariable(1, 32, 0)
	w3 = goraph.NewRandomVariable(32, 1, goraph.NewKaimingNormalInit(32))
	b3 = goraph.NewConstVariable(1, 1, 0)

	parameters = []*goraph.VariableNode{pw, w1, b1, w2, b2, w3, b3}
)

func buildGraph() (input, target *goraph.VariableNode, output, loss goraph.Node) {
	input = goraph.NewConstVariable(1, 2, 0)
	target = goraph.NewConstVariable(1, 1, 0)
	output = goraph.Add(input, pw)

	output = goraph.Multi(output, w1)
	output = goraph.Add(output, b1)
	output = goraph.ReLu(output)

	output = goraph.Multi(output, w2)
	output = goraph.Add(output, b2)
	output = goraph.ReLu(output)

	output = goraph.Multi(output, w3)
	output = goraph.Add(output, b3)

	loss = goraph.MSELoss(output, target)
	return
}
func main() {
	optimizer := goraph.NewSGDOptimizer(parameters, 0.001, 0.9)
	nn := goraph.NewNeuralNetwork(buildGraph, optimizer)
	{
		var inputData, targetData [][]float64
		for i := range 10 {
			for j := range 9 {
				inputData = append(inputData, []float64{float64(i), float64(j + 1)})
				targetData = append(targetData, []float64{float64(i) / float64(j+1)})
			}
		}
		for epoch := range 20000 {
			lossValue := nn.Train(inputData, targetData, 2)
			fmt.Printf("Epoch: %v, Loss: %.10f\n", epoch, lossValue)
		}
	}
	{
		var inputData, targetData [][]float64
		inputData = [][]float64{{3, 2}, {7, 2}, {3, 8}, {4, 2}}
		targetData = [][]float64{{1.5}, {3.5}, {0.375}, {2}}
		lossValue, outputData := nn.Evaluate(inputData, targetData)
		for i, v := range outputData {
			fmt.Printf("Output: %v, Target: %v\n", v[0], targetData[i])
		}
		fmt.Printf("Loss: %.10f\n", lossValue)
	}
}
