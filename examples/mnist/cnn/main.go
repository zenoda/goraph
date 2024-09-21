package main

import (
	"fmt"
	"goraph"
	"math/rand/v2"
	"os"
)

func main() {
	input := goraph.NewConstVariable(28, 28, 0, "input")
	kernel := goraph.NewRandomVariable(2, 2, rand.Float64, "kernel")
	bk := goraph.NewConstVariable(1, 1, 0.001, "bk")
	w1 := goraph.NewRandomVariable(14*14, 10, rand.Float64, "w1")
	b1 := goraph.NewConstVariable(1, 10, 0.001, "b1")
	target := goraph.NewConstVariable(1, 10, 0, "target")
	var output goraph.Node
	output = goraph.Conv(input, kernel, 1)
	cbb1 := goraph.NewConstVariable(28, 1, 1, "cbb1")
	cbb2 := goraph.NewConstVariable(1, 28, 1, "cbb2")
	cbb := goraph.Multi(cbb1, bk)
	cbb = goraph.Multi(cbb, cbb2)
	output = goraph.Add(output, cbb)
	output = goraph.Tanh(output)
	output = goraph.Pool(output, 2, 2, 2)
	output = goraph.Reshape(output, 1, 14*14)
	output = goraph.Multi(output, w1)
	output = goraph.Add(output, b1)
	output = goraph.Softmax(output)

	var loss goraph.Node
	loss = goraph.CrossEntropyLoss(output, target)

	parameters := []*goraph.VariableNode{kernel, bk, w1, b1}
	optimizer := goraph.NewSGDOptimizer(parameters, 0.001, 0.9)

	model := goraph.NewModel(parameters, nil)
	model.Load("model.json")
	trainInputs, trainTargets := readSamples("train")
	for epoch := range 10 {
		lossVal := goraph.NewConstMatrix(1, 1, 0)
		for i, inputData := range trainInputs {
			input.Value = inputData
			target.Value = trainTargets[i]
			lossVal = lossVal.Add(loss.Forward())
			loss.Backward(nil)
			optimizer.Step()
			loss.Reset()
		}
		fmt.Printf("Epoch %d, Loss: %v\n", epoch, lossVal.Scale(1/float64(len(trainInputs))).Data)
	}
	model.Save("model.json")

	testInputs, testTargets := readSamples("test")
	{
		lossValue := goraph.NewConstMatrix(1, 1, 0)
		rate := 0.0
		for i := range testInputs {
			input.Value = testInputs[i]
			target.Value = testTargets[i]
			result := output.Forward()
			maxIdx := 0
			maxVal := 0.0
			for j, v := range result.Data {
				if v > maxVal {
					maxVal = v
					maxIdx = j
				}
			}
			if testTargets[i].Data[maxIdx] == 1.0 {
				rate += 1.0
			}
			lossValue = lossValue.Add(loss.Forward())
			loss.Reset()
		}
		fmt.Printf("Test, Success ratio: %f\n", rate/float64(len(testInputs)))
		fmt.Printf("Test, Loss: %v\n", lossValue.Scale(1/float64(len(testInputs))))
	}
}

func readSamples(sampleType string) (inputs, targets []*goraph.Matrix) {
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
	for i, img := range imgs {
		var inputData, labelData []float64
		for k := range img {
			inputData = append(inputData, float64(img[k])/256*0.9+0.1)
		}
		labelData = make([]float64, 10)
		labelData[labels[i]] = 1.0
		inputs = append(inputs, goraph.NewMatrix(28, 28, inputData))
		targets = append(targets, goraph.NewMatrix(1, 10, labelData))
	}
	return
}
