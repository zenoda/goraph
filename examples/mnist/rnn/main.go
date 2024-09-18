package main

import (
	"fmt"
	"goraph"
	"math/rand/v2"
	"os"
)

func NewRandFunc(dim int) func() float64 {
	return func() float64 {
		return rand.Float64() / float64(dim)
	}
}
func main() {
	input := goraph.NewConstVariable(28, 28, 0, "input")
	w1 := goraph.NewRandomVariable(28, 56, NewRandFunc(28), "w1")
	b1 := goraph.NewConstVariable(1, 56, 0.001, "b1")
	rw := goraph.NewRandomVariable(56, 56, NewRandFunc(56), "rw")
	w2 := goraph.NewRandomVariable(56, 10, NewRandFunc(56), "w2")
	b2 := goraph.NewConstVariable(1, 10, 0.001, "b2")
	target := goraph.NewConstVariable(1, 10, 0, "target")

	var output goraph.Node
	for i := range 28 {
		if i == 0 {
			output = goraph.NewConstVariable(1, 56, 0, "hidden")
		}
		hi := goraph.Multi(goraph.RowSlice(input, i, i+1), w1)
		ho := goraph.Multi(output, rw)
		output = goraph.Add(hi, ho)
		output = goraph.Add(output, b1)
		output = goraph.Tanh(output)
	}
	output = goraph.Multi(output, w2)
	output = goraph.Add(output, b2)
	output = goraph.Softmax(output)

	var loss goraph.Node
	loss = goraph.CrossEntropyLoss(output, target)

	parameters := []*goraph.VariableNode{w1, b1, rw, w2, b2}
	//optimizer := goraph.NewAdamOptimizer(parameters, 0.001, 0.9, 0.999, 1e-8)
	optimizer := goraph.NewSGDOptimizer(parameters, 0.0001, 0.9)

	model := goraph.NewModel(parameters, nil)
	model.Load("rnn.json")
	trainInputs, trainTargets := readSamples("train")
	for epoch := range 100 {
		lossValue := goraph.NewConstMatrix(1, 1, 0)
		for i := range trainInputs {
			input.Value = trainInputs[i]
			target.Value = trainTargets[i]
			lossValue = lossValue.Add(loss.Forward())
			loss.Backward(nil)
			optimizer.Step()
			loss.Reset()
			fmt.Printf("epoch %d: %d%%, loss value: %f\r", epoch, (i+1)*100/len(trainInputs), lossValue.Scale(1/float64(i+1)).Data)
		}
		fmt.Printf("Epoch: %d, Loss: %v\n", epoch, lossValue.Scale(1/float64(len(trainInputs))))
	}
	model.Save("rnn.json")

	testInputs, testTargets := readSamples("test")
	{
		lossValue := goraph.NewConstMatrix(1, 1, 0)
		rate := 0.0
		for i := range testInputs {
			input.Value = testInputs[i]
			target.Value = testTargets[i]
			result := output.Forward()
			maxIdx := 0
			max := 0.0
			for j, v := range result.Data {
				if v > max {
					max = v
					maxIdx = j
				}
			}
			if testTargets[i].Data[maxIdx] == 1.0 {
				rate += 1.0
			}
			lossValue = lossValue.Add(loss.Forward())
			loss.Reset()
		}
		fmt.Printf("Test, Success ratio: %d\n", rate/float64(len(testInputs)))
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
