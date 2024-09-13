package goraph

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
)

type Model struct {
	Parameters []*VariableNode `json:"parameters"`
	Scaler     Scaler          `json:"scaler"`
}

func NewModel(parameters []*VariableNode, scaler Scaler) *Model {
	return &Model{
		Parameters: parameters,
		Scaler:     scaler,
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
	log.Print("Model file not found")
	return nil
}

func (m *Model) String() string {
	return fmt.Sprintf("Parameters: %v, Scaler: %v", m.Parameters, m.Scaler)
}
