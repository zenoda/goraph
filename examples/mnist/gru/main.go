package main

import (
	"goraph"
	"os"
)

func main() {

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
