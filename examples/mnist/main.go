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
	input := goraph.NewConstVariable(10, 784, 0, "input")
	w1 := goraph.NewRandomVariable(784, 10, NewRandFunc(784), "w1")
	b1 := goraph.NewConstVariable(1, 10, 0.1, "b1")
	w2 := goraph.NewRandomVariable(10, 1, NewRandFunc(10), "w2")
	b2 := goraph.NewConstVariable(1, 1, 0.1, "b2")
	target := goraph.NewConstVariable(10, 1, 0, "target")

	var output goraph.Node
	output = goraph.Multi(input, w1)
	output = goraph.Add(output, goraph.Multi(goraph.NewConstVariable(10, 1, 1, "bb1"), b1))
	output = goraph.ReLu(output)

	output = goraph.Multi(output, w2)
	output = goraph.Add(output, goraph.Multi(goraph.NewConstVariable(10, 1, 1, "bb2"), b2))
	output = goraph.Sigmoid(output)

	var loss goraph.Node
	loss = goraph.MSELoss(output, target)

	parameters := []*goraph.VariableNode{w1, b1, w2, b2}
	optimizer := goraph.NewSGDOptimizer(parameters, 0.1, 0)

	model := goraph.NewModel(parameters, nil)
	model.Load("model.json")
	inputs, targets := readSamples("train", 10)
	for epoch := range 30 {
		lossValue := goraph.NewConstMatrix(1, 1, 0)
		for i := range inputs {
			input.Value = inputs[i]
			target.Value = targets[i]
			lossValue = lossValue.Add(loss.Forward())
			loss.Backward(nil)
			optimizer.Step()
			//fmt.Printf("Output:%v, Target:%v\n", output.Forward(), target.Value)
			loss.Reset()
		}
		fmt.Printf("Epoch: %d, Loss: %v\n", epoch, lossValue.Scale(1/float64(len(inputs))))
	}
	model.Save("model.json")

	inputs, targets = readSamples("test", 10)
	lossValue := goraph.NewConstMatrix(1, 1, 0)
	for i := range inputs {
		input.Value = inputs[i]
		target.Value = targets[i]
		lossValue = lossValue.Add(loss.Forward())
		fmt.Printf("Output:%v, Target:%v\n", output.Forward(), target.Value)
		loss.Reset()
	}
	fmt.Printf("Test, Loss: %v\n", lossValue.Scale(1/float64(len(inputs))))
}

func readSamples(sampleType string, batchSize int) (inputs, targets []*goraph.Matrix) {
	var imgFilePath, labelFilePath string
	switch sampleType {
	case "train":
		imgFilePath = "dataset/train-images-idx3-ubyte"
		labelFilePath = "dataset/train-labels-idx1-ubyte"
	case "test":
		imgFilePath = "dataset/t10k-images-idx3-ubyte"
		labelFilePath = "dataset/t10k-labels-idx1-ubyte"
	}
	imgs, err := readImageFile(os.Open(imgFilePath))
	if err != nil {
		panic(err)
	}
	labels, err := readLabelFile(os.Open(labelFilePath))
	if err != nil {
		panic(err)
	}
	for i := 0; i*batchSize < len(imgs); i++ {
		var inputData, labelData []float64
		for j := 0; j < batchSize; j++ {
			for k := range imgs[i*batchSize+j] {
				inputData = append(inputData, float64(imgs[i*batchSize+j][k])/256*0.9+0.1)
			}
			labelData = append(labelData, float64(labels[i*batchSize+j])/10)
		}
		inputs = append(inputs, goraph.NewMatrix(batchSize, 784, inputData))
		targets = append(targets, goraph.NewMatrix(batchSize, 1, labelData))
	}
	return
}
