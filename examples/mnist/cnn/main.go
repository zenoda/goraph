package main

import (
	"fmt"
	"goraph"
	"math/rand/v2"
	"os"
)

func main() {
	kernel := goraph.NewRandomVariable(2, 2, rand.Float64)
	bk := goraph.NewConstVariable(1, 1, 0.001)
	w1 := goraph.NewRandomVariable(14*14, 10, rand.Float64)
	b1 := goraph.NewConstVariable(1, 10, 0.001)

	parameters := []*goraph.VariableNode{kernel, bk, w1, b1}
	optimizer := goraph.NewSGDOptimizer(parameters, 0.001, 0.9)

	builder := func() (input, target *goraph.VariableNode, output, loss goraph.Node) {
		input = goraph.NewConstVariable(28, 28, 0)
		target = goraph.NewConstVariable(1, 10, 0)
		output = goraph.Conv(input, kernel, 1)
		cbb1 := goraph.NewConstVariable(28, 1, 1)
		cbb2 := goraph.NewConstVariable(1, 28, 1)
		cbb := goraph.Multi(cbb1, bk)
		cbb = goraph.Multi(cbb, cbb2)
		output = goraph.Add(output, cbb)
		output = goraph.Tanh(output)
		output = goraph.Pool(output, 2, 2, 2)
		output = goraph.Reshape(output, 1, 14*14)
		output = goraph.Multi(output, w1)
		output = goraph.Add(output, b1)
		output = goraph.Softmax(output)
		loss = goraph.CrossEntropyLoss(output, target)
		return
	}

	nn := goraph.NewNeuralNetwork(builder, optimizer)

	model := goraph.NewModel(parameters, nil)
	model.Load("model.json")

	{
		inputData, targetData := readSamples("train")
		for epoch := range 10 {
			lossVal := nn.Train(inputData, targetData, 30)
			fmt.Printf("Epoch %d, Loss: %v\n", epoch, lossVal)
		}
	}
	model.Save("model.json")

	{
		inputData, targetData := readSamples("test")
		lossVal := nn.Evaluate(inputData, targetData)
		fmt.Printf("Test, Loss: %v\n", lossVal)
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
