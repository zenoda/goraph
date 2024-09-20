package main

import (
	"fmt"
	"goraph"
	"math/rand/v2"
	"os"
	"reflect"
)

func NewRandFunc(dim int) func() float64 {
	return func() float64 {
		return rand.Float64() / float64(dim)
	}
}
func main() {
	input := goraph.NewConstVariable(28, 28, 0, "input")
	wz := goraph.NewRandomVariable(78, 50, NewRandFunc(78), "wz")
	bz := goraph.NewConstVariable(1, 50, 0.001, "bz")
	wr := goraph.NewRandomVariable(78, 50, NewRandFunc(78), "wr")
	br := goraph.NewConstVariable(1, 50, 0.001, "br")
	wh := goraph.NewRandomVariable(78, 50, NewRandFunc(78), "wh")
	bh := goraph.NewConstVariable(1, 50, 0.001, "bh")

	w2 := goraph.NewRandomVariable(50, 10, NewRandFunc(50), "w2")
	b2 := goraph.NewConstVariable(1, 10, 0.001, "b2")
	target := goraph.NewConstVariable(1, 10, 0, "target")

	var output, zt, ztb, rt, htHat goraph.Node
	ztb = goraph.NewConstVariable(1, 50, 1, "ztb")
	for i := range 28 {
		if i == 0 {
			output = goraph.NewConstVariable(1, 50, 0, "hidden")
		}
		rowInput := goraph.RowSlice(input, i, i+1)
		zt = goraph.HConcat(output, rowInput)
		zt = goraph.Multi(zt, wz)
		zt = goraph.Add(zt, bz)
		zt = goraph.Sigmoid(zt)

		rt = goraph.HConcat(output, rowInput)
		rt = goraph.Multi(rt, wr)
		rt = goraph.Add(rt, br)
		rt = goraph.Sigmoid(rt)

		htHat = goraph.MultiElement(output, rt)
		htHat = goraph.HConcat(htHat, rowInput)
		htHat = goraph.Multi(htHat, wh)
		htHat = goraph.Add(htHat, bh)
		htHat = goraph.Tanh(htHat)

		output = goraph.MultiElement(output, zt)
		output = goraph.Add(output, goraph.MultiElement(htHat, goraph.Sub(ztb, zt)))
		output = goraph.GradThreshold(output, 1e-4)
	}
	output = goraph.Multi(output, w2)
	output = goraph.Add(output, b2)
	output = goraph.Softmax(output)

	var loss goraph.Node
	loss = goraph.CrossEntropyLoss(output, target)
	//drawGraph(loss, 0)
	parameters := []*goraph.VariableNode{wz, bz, wr, br, wh, bh, w2, b2}
	//optimizer := goraph.NewAdamOptimizer(parameters, 0.001, 0.9, 0.999, 1e-8)
	optimizer := goraph.NewSGDOptimizer(parameters, 0.01, 0.9)

	model := goraph.NewModel(parameters, nil)
	model.Load("rnn.json")
	trainInputs, trainTargets := readSamples("train")
	for epoch := range 50 {
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
		fmt.Printf("Test, Success ratio: %f\n", rate/float64(len(testInputs)))
		fmt.Printf("Test, Loss: %v\n", lossValue.Scale(1/float64(len(testInputs))))
	}
}

func drawGraph(node goraph.Node, level int) {
	//if level > 60 {
	//	panic("level too large")
	//}
	for range level {
		fmt.Print("-")
	}
	t := reflect.TypeOf(node)
	fmt.Println(t)
	if node.GetDeps() != nil {
		for _, dep := range node.GetDeps() {
			drawGraph(dep, level+1)
		}
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
