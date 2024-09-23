package main

import (
	"fmt"
	"goraph"
	"math/rand/v2"
	"os"
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
	optimizer := goraph.NewSGDOptimizer(parameters, 0.001, 0)

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
		inputData, targetData := readSamples("train")
		for epoch := range 30 {
			lossValue := nn.Train(inputData, targetData, 20)
			fmt.Printf("Epoch: %d, loss: %f\n", epoch, lossValue)
		}
	}
	model.Save("model.json")
	{
		inputData, targetData := readSamples("test")
		lossValue := nn.Evaluate(inputData, targetData)
		fmt.Printf("Test, Loss: %v\n", lossValue)
	}
}

func readSamples(sampleType string) (inputData, targetData [][]float64) {
	var imgFilePath, labelFilePath string
	switch sampleType {
	case "train":
		imgFilePath = "../dataset/train-images-idx3-ubyte"
		labelFilePath = "../dataset/train-labels-idx1-ubyte"
	case "test":
		imgFilePath = "../dataset/t10k-images-idx3-ubyte"
		labelFilePath = "../dataset/t10k-labels-idx1-ubyte"
	}
	imgs, err := readImageFile(os.Open(imgFilePath))
	if err != nil {
		panic(err)
	}
	labels, err := readLabelFile(os.Open(labelFilePath))
	if err != nil {
		panic(err)
	}
	for i := range imgs {
		var imgData []float64
		var labelData = make([]float64, 10)
		for _, pixel := range imgs[i] {
			imgData = append(imgData, float64(pixel)/256*0.9+0.1)
		}
		labelData[labels[i]] = 1
		inputData = append(inputData, imgData)
		targetData = append(targetData, labelData)
	}
	return
}
