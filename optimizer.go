package goraph

type Optimizer interface {
	Step()
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
func (opt *SGDOptimizer) Step() {
	for i, p := range opt.Parameters {
		opt.Velocity[i] = opt.Velocity[i].Scale(opt.Momentum).Add(p.Gradient.Scale(1 - opt.Momentum))
		p.Value = p.Value.Sub(opt.Velocity[i].Scale(opt.LearningRate))
	}
}
