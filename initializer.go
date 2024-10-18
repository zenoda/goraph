package goraph

import (
	"math"
	"math/rand/v2"
)

func NewXavierNormalInit(fanIn, fanOut int) func() float64 {
	return func() float64 {
		return (rand.Float64()*2 - 1) * math.Sqrt(2.0/float64(fanIn+fanOut))
	}
}

func NewXavierUniformInit(fanIn, fanOut int) func() float64 {
	return func() float64 {
		return (rand.Float64()*2 - 1) * math.Sqrt(6.0/float64(fanIn+fanOut))
	}
}

func NewKaimingNormalInit(fanIn int) func() float64 {
	return func() float64 {
		return (rand.Float64()*2 - 1) * math.Sqrt(2.0/float64(fanIn))
	}
}
