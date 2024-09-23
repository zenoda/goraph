package goraph

import "math"

type Optimizer interface {
	Step(batchSize int)
	Reset()
}

type SGDOptimizer struct {
	LearningRate float64
	Momentum     float64
	Velocity     []*Matrix
	Parameters   []*VariableNode
}

func NewSGDOptimizer(parameters []*VariableNode, learningRate, momentum float64) *SGDOptimizer {
	velocity := make([]*Matrix, len(parameters))
	for i := range velocity {
		velocity[i] = NewConstMatrix(parameters[i].Value.Rows, parameters[i].Value.Cols, 0)
	}
	return &SGDOptimizer{
		LearningRate: learningRate,
		Momentum:     momentum,
		Velocity:     velocity,
		Parameters:   parameters,
	}
}
func (opt *SGDOptimizer) Step(batchSize int) {
	for i, p := range opt.Parameters {
		grad := p.Gradient.Scale(1 / float64(batchSize))
		opt.Velocity[i] = opt.Velocity[i].Scale(opt.Momentum).Add(grad.Scale(1 - opt.Momentum))
		p.Value = p.Value.Sub(opt.Velocity[i].Scale(opt.LearningRate))
	}
}

func (opt *SGDOptimizer) Reset() {
	for i := range opt.Velocity {
		opt.Velocity[i] = NewConstMatrix(opt.Velocity[i].Rows, opt.Velocity[i].Cols, 0)
	}
}

type AdamOptimizer struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	M            []*Matrix
	V            []*Matrix
	T            int
	Eps          float64
	Parameters   []*VariableNode
}

func NewAdamOptimizer(parameters []*VariableNode, learningRate, beta1, beta2, eps float64) *AdamOptimizer {
	m := make([]*Matrix, len(parameters))
	for i := range m {
		m[i] = NewConstMatrix(parameters[i].Value.Rows, parameters[i].Value.Cols, 0)
	}
	v := make([]*Matrix, len(parameters))
	for i := range v {
		v[i] = NewConstMatrix(parameters[i].Value.Rows, parameters[i].Value.Cols, 0)
	}
	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Eps:          eps,
		M:            m,
		V:            v,
		T:            1,
		Parameters:   parameters,
	}
}

func (opt *AdamOptimizer) Step(batchSize int) {
	for i, p := range opt.Parameters {
		grad := p.Gradient.Scale(1 / float64(batchSize))
		for j := range grad.Data {
			m := opt.Beta1*opt.M[i].Data[j] + (1-opt.Beta1)*grad.Data[j]
			v := opt.Beta2*opt.V[i].Data[j] + (1-opt.Beta2)*math.Pow(grad.Data[j], 2)
			mHat := m / (1 - math.Pow(opt.Beta1, float64(opt.T)))
			vHat := v / (1 - math.Pow(opt.Beta2, float64(opt.T)))
			update := opt.LearningRate * mHat / (math.Sqrt(vHat) + opt.Eps)
			opt.Parameters[i].Value.Data[j] -= update
			opt.M[i].Data[j] = m
			opt.V[i].Data[j] = v
		}
	}
	opt.T++
}

func (opt *AdamOptimizer) Reset() {
	for i := range opt.M {
		opt.M[i] = NewConstMatrix(opt.M[i].Rows, opt.M[i].Cols, 0)
	}
	for i := range opt.V {
		opt.V[i] = NewConstMatrix(opt.V[i].Rows, opt.V[i].Cols, 0)
	}
	opt.T = 1
}
