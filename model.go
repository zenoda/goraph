package goraph

import (
	"encoding/json"
	"fmt"
	"os"
)

type Model struct {
	Parameters    []*VariableNode `json:"parameters"`
	InputScalers  []Scaler        `json:"input_scalers"`
	TargetScalers []Scaler        `json:"target_scalers"`
}

func NewModel(parameters []*VariableNode, inputScalers, targetScalers []Scaler) *Model {
	return &Model{
		Parameters:    parameters,
		InputScalers:  inputScalers,
		TargetScalers: targetScalers,
	}
}

func (m *Model) Save(filePath string) error {
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0660)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	return encoder.Encode(m)
}

func (m *Model) Load(filePath string) error {
	_, err := os.Stat(filePath)
	if err == nil {
		file, err := os.Open(filePath)
		if err != nil {
			return err
		}
		defer file.Close()
		decoder := json.NewDecoder(file)
		return decoder.Decode(m)
	}
	return err
}

func (m *Model) String() string {
	return fmt.Sprintf("Parameters: %v, InputScalers: %v, TargetScalers: %v", m.Parameters, m.InputScalers, m.TargetScalers)
}
