package goraph

import (
	"sync"
)

type NeuralNetwork struct {
	buildFunc func() (input, target *VariableNode, output, loss Node)
	optimizer Optimizer
}

func (nn *NeuralNetwork) Train(inputData, targetData [][]float64, batchSize int) (lossValue float64) {
	inputs := make([]*VariableNode, batchSize)
	targets := make([]*VariableNode, batchSize)
	losses := make([]Node, batchSize)
	for i := 0; i < batchSize; i++ {
		inputs[i], targets[i], _, losses[i] = nn.buildFunc()
	}
	for i := 0; i*batchSize < len(inputData); i++ {
		realBatchSize := min(len(inputData)-i*batchSize, batchSize)
		var wg sync.WaitGroup
		var mu sync.Mutex
		var lossBatch float64
		for j := 0; j < realBatchSize; j++ {
			wg.Add(1)
			go func(batch, idx, batchSize int) {
				inputs[idx].Value = NewMatrix(inputs[idx].Value.Rows, inputs[idx].Value.Cols, inputData[batch*batchSize+idx])
				targets[idx].Value = NewMatrix(targets[idx].Value.Rows, targets[idx].Value.Cols, targetData[batch*batchSize+idx])
				mu.Lock()
				lossBatch += losses[idx].Forward().Data[0]
				mu.Unlock()
				losses[idx].Backward(nil)
				wg.Done()
			}(i, j, batchSize)
		}
		wg.Wait()
		lossValue += lossBatch / float64(realBatchSize)
		nn.optimizer.Step(realBatchSize)
		for j := 0; j < realBatchSize; j++ {
			losses[j].Reset()
		}
	}
	lossValue /= float64(len(inputData))
	return
}

func (nn *NeuralNetwork) Evaluate(inputData, targetData [][]float64) (lossValue float64, outputData [][]float64) {
	input, target, output, loss := nn.buildFunc()
	outputData = make([][]float64, len(inputData))
	for i := range inputData {
		input.Value = NewMatrix(input.Value.Rows, input.Value.Cols, inputData[i])
		target.Value = NewMatrix(target.Value.Rows, target.Value.Cols, targetData[i])
		outputData[i] = output.Forward().Data
		lossValue += loss.Forward().Data[0]
		loss.Reset()
	}
	lossValue /= float64(len(inputData))
	return
}
func (nn *NeuralNetwork) Predict(inputData []float64) (outputData []float64) {
	input, _, output, _ := nn.buildFunc()
	input.Value = NewMatrix(input.Value.Rows, input.Value.Cols, inputData)
	outputData = output.Forward().Data
	return
}

func NewNeuralNetwork(
	buildFunc func() (input, target *VariableNode, output, loss Node),
	optimizer Optimizer) *NeuralNetwork {
	return &NeuralNetwork{
		buildFunc: buildFunc,
		optimizer: optimizer,
	}
}
