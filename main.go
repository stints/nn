package main

import (
	"fmt"
	"math"

	"github.com/stints/matrix"
)

var (
	inputCount   = 2
	hiddenCount  = 2
	outputCount  = 1
	learningRate = 0.2
	epoch        = 100000

	trainingData = [][][]float64{
		{{0, 0}, {0}},
		{{1, 1}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
	}

	hiddenWeights = matrix.New(hiddenCount, inputCount)
	hiddenBias    = matrix.New(hiddenCount, 1)
	outputWeights = matrix.New(outputCount, hiddenCount)
	outputBias    = matrix.New(outputCount, 1)
)

func init() {
	hiddenWeights.Randomize(-3, 3)
	hiddenBias.Randomize(-3, 3)
	outputWeights.Randomize(-3, 3)
	outputBias.Randomize(-3, 3)
}

func main() {
	fmt.Println("Starting: ", hiddenWeights, hiddenBias, outputWeights, outputBias)
	for _, data := range trainingData {
		// Get Inputs and Target
		input := matrix.FromArray(data[0], true)
		target := matrix.FromArray(data[1], true)

		// Feed Forward
		hidden := matrix.Multiply(hiddenWeights, input).Add(hiddenBias).Map(sigmoid)
		output := matrix.Multiply(outputWeights, hidden).Add(outputBias).Map(sigmoid)

		fmt.Println("Input: ", input, "Target: ", target, "Guess: ", output)
	}
	fmt.Println("-----")

	for n := 0; n < epoch; n++ {
		for _, data := range trainingData {
			// Get Inputs and Target
			input := matrix.FromArray(data[0], true)
			target := matrix.FromArray(data[1], true)

			// Feed Forward
			hidden := matrix.Multiply(hiddenWeights, input).Add(hiddenBias).Map(sigmoid)
			output := matrix.Multiply(outputWeights, hidden).Add(outputBias).Map(sigmoid)

			// Back Prop
			// Output Weights and Bias
			outputError := matrix.Subtract(output, target)
			outputGradient := matrix.Map(output, sigmoidPrime).Hadamard(outputError)

			//fmt.Println("Hidden: ", hidden, "HiddenT: ", matrix.Transpose(hidden), "Output Gradient: ", outputGradient)
			outputWeightDelta := matrix.Transpose(hidden).Scalar(outputGradient.Get(1, 1))
			outputBiasDelta := outputGradient

			outputWeights.Subtract(outputWeightDelta.Scalar(learningRate))
			outputBias.Subtract(outputBiasDelta.Scalar(learningRate))

			// Hidden Weights and Bias
			hiddenGradient := matrix.Map(hidden, sigmoidPrime).Hadamard(matrix.Transpose(outputWeights).Scalar(outputGradient.Get(1, 1)))
			hiddenWeightDelta := matrix.Multiply(hiddenGradient, matrix.Transpose(input))
			hiddenBiasDelta := hiddenGradient

			hiddenWeights.Subtract(hiddenWeightDelta.Scalar(learningRate))
			hiddenBias.Subtract(hiddenBiasDelta.Scalar(learningRate))
		}
	}
	fmt.Println("Finished: ", hiddenWeights, hiddenBias, outputWeights, outputBias)

	for _, data := range trainingData {
		// Get Inputs and Target
		input := matrix.FromArray(data[0], true)
		target := matrix.FromArray(data[1], true)

		// Feed Forward
		hidden := matrix.Multiply(hiddenWeights, input).Add(hiddenBias).Map(sigmoid)
		output := matrix.Multiply(outputWeights, hidden).Add(outputBias).Map(sigmoid)

		fmt.Println("Input: ", input, "Target: ", target, "Guess: ", output)
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}
