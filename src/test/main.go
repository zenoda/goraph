package main

import (
	"fmt"
	"goraph/src/goraph"
)

func main() {
	m1 := goraph.NewMatrix(1, 2, []float64{1, 2})
	w1 := goraph.NewMatrix(2, 5, []float64{1, 2, 3, 4, 5, 2, 1, 0, 2, 2})
	b1 := goraph.NewMatrix(1, 5, []float64{0.5, 0.8, 0.3, 0.1, 0.2})
	v1 := goraph.NewVariable(m1)
	v2 := goraph.NewVariable(w1)
	v3 := goraph.NewVariable(b1)
	multi := goraph.NewMulti(v1, v2)
	add := goraph.NewAdd(multi, v3)
	sig := goraph.NewSigmoid(add)
	fmt.Println(sig.Forward())
	sig.Backward(goraph.NewMatrix(1, 5, []float64{0.1, 0.2, 0.3, 0.4, 0.5}))
	fmt.Println(v3.Gradient)
	sig.Reset()
	fmt.Println(v3.Gradient)
}
