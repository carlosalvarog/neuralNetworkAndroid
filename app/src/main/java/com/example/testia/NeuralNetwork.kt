package com.example.testia

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class NeuralNetwork(private val listener: NeuralNetworkListener? = null) {
    private val inputSize = 3
    private val layer1Size = 3
    private val layer2Size = 2


    //we set up as random all the weights and biases (from 0, to 1)
    private val weightsLayer1 = Array(layer1Size) { FloatArray(inputSize) { (Math.random().toFloat()) } }
    private val biasesLayer1 = FloatArray(layer1Size) { (Math.random().toFloat()) }

    private val weightsLayer2 = Array(layer2Size) { FloatArray(layer1Size) { (Math.random().toFloat()) } }
    private val biasesLayer2 = FloatArray(layer2Size) { (Math.random().toFloat()) }

    private val weightsOutput = FloatArray(layer2Size) { (Math.random().toFloat()) }
    private var biasOutput = (Math.random().toFloat())


    //the outputs 'a' for each neuron
    val layer1Output = FloatArray(layer1Size)
    val layer2Output = FloatArray(layer2Size)
    var output = 0f;

    //hyperparameter
    var learningRate = 0.001f;

    //for each of our samples, our 'predicted' values for each sample, and the actual values of each sample
    var predicted: MutableList<Float> = mutableListOf()
    var actual: MutableList<Float> = mutableListOf()

    //target
    var target = 0f;

    fun forward(inputs: FloatArray): Float {


        for (i in 0 until layer1Size) {
            var sum = 0f
            for (j in 0 until inputSize) {
                sum += weightsLayer1[i][j] * inputs[j]
            }
            sum += biasesLayer1[i]
            layer1Output[i] = relu(sum)
        }

        // Calculate layer 2 activations, using our input a1 (layer1Output) and layer2Output as a2

        for (i in 0 until layer2Size) {
            var sum = 0f
            for (j in 0 until layer1Size) {
                sum += weightsLayer2[i][j] * layer1Output[j]
            }
            sum += biasesLayer2[i]
            layer2Output[i] = relu(sum)
        }

        // Calculate final output, we only have one row so we only have a vector operation, no matrix, that's why we can save ourselves from making a i,j matrix operation
        var layer3Output = biasOutput
        for (j in 0 until layer2Size) {
            layer3Output += weightsOutput[j] * layer2Output[j]
        }
        return relu(layer3Output)
    }

    private fun calculateOutputError(target: Float): Float {
        return (output - target) * reluDerivative(output)
    }

    // 3. Backpropagate to the Hidden Layers
    private fun backpropagateHiddenLayers(outputError: Float): Pair<FloatArray, FloatArray> {
        // Errors at layer 2 (backpropagated from output)
        val layer2Errors = FloatArray(layer2Size)
        for (i in 0 until layer2Size) {
            layer2Errors[i] = outputError * reluDerivative(layer2Output[i]) * weightsOutput[i] // Error for layer 2 neurons
        }

        // Errors at layer 1 (hidden layer)
        val layer1Errors = FloatArray(layer1Size)
        for (i in 0 until layer1Size) {
            var sum = 0f
            for (j in 0 until layer2Size) {
                sum += layer2Errors[j] * weightsLayer2[j][i]
            }
            layer1Errors[i] = sum * reluDerivative(layer1Output[i]) // Error for layer 1 neurons
        }

        return Pair(layer1Errors, layer2Errors)
    }

    //we receive a dataset that is a list of pairs (our input (a float array) and our target
    //We will pass 'epochs' times through dataset, doing mean gradients of size batchSize samples
    fun train(dtst: List<Pair<FloatArray, Float>>, epochs: Int, batchSize: Int) {

        //we will be using coroutines for our training so we can update our graph after each epoch
        CoroutineScope(Dispatchers.Default).launch {
            var dataset = dtst;

            // The number of times we cycle through our entire dataset
            for (epoch in 1..epochs) {
                println("Epoch $epoch")

                //predicted and actual are a list that is filled every epoch with our predicted value and the actual value of the dew point
                predicted = mutableListOf()
                actual = mutableListOf()

                // Shuffle the dataset at the start of each epoch
                dataset = dataset.shuffled()

                // Process in batches of size batchSize
                for (batch in dataset.chunked(batchSize)) {

                    // Initialize accumulators for gradients (for doing the mean)
                    val weightOutputGradientAccumulator = FloatArray(layer2Size)
                    var biasOutputGradientAccumulator = 0f

                    val weightsLayer2GradientAccumulator = Array(layer2Size) { FloatArray(layer1Size) }
                    val biasesLayer2GradientAccumulator = FloatArray(layer2Size)

                    val weightsLayer1GradientAccumulator = Array(layer1Size) { FloatArray(inputSize) }
                    val biasesLayer1GradientAccumulator = FloatArray(layer1Size)

                    // Process each sample in the batch
                    for ((inputs, targetSample) in batch) {
                        // Forward pass
                        output = forward(inputs)
                        target = targetSample
                        predicted.add(output)
                        actual.add(target)

                        // calculate the error at the output
                        val outputError = calculateOutputError(target)
                        //and then backpropagate the error to calculate the errors in other neurons of other layers
                        val (layer1Errors, layer2Errors) = backpropagateHiddenLayers(outputError)

                        // Accumulate gradients for Output Layer
                        for (i in 0 until layer2Size) {
                            weightOutputGradientAccumulator[i] += outputError * layer2Output[i]
                        }
                        biasOutputGradientAccumulator += outputError

                        // Accumulate gradients for Layer 2
                        for (i in 0 until layer2Size) {
                            for (j in 0 until layer1Size) {
                                weightsLayer2GradientAccumulator[i][j] += layer2Errors[i] * layer1Output[j]
                            }
                            biasesLayer2GradientAccumulator[i] += layer2Errors[i]
                        }

                        // Accumulate gradients for Layer 1
                        for (i in 0 until layer1Size) {
                            for (j in 0 until inputSize) {
                                weightsLayer1GradientAccumulator[i][j] += layer1Errors[i] * inputs[j]
                            }
                            biasesLayer1GradientAccumulator[i] += layer1Errors[i]
                        }
                    }

                    // Update weights and biases for Output Layer, now we divide by our batch size so we have a mean
                    for (i in 0 until layer2Size) {
                        weightsOutput[i] -= learningRate * (weightOutputGradientAccumulator[i] / batch.size)
                    }
                    biasOutput -= learningRate * (biasOutputGradientAccumulator / batch.size)

                    // Update weights and biases for Layer 2
                    for (i in 0 until layer2Size) {
                        for (j in 0 until layer1Size) {
                            weightsLayer2[i][j] -= learningRate * (weightsLayer2GradientAccumulator[i][j] / batch.size)
                        }
                        biasesLayer2[i] -= learningRate * (biasesLayer2GradientAccumulator[i] / batch.size)
                    }

                    // Update weights and biases for Layer 1
                    for (i in 0 until layer1Size) {
                        for (j in 0 until inputSize) {
                            weightsLayer1[i][j] -= learningRate * (weightsLayer1GradientAccumulator[i][j] / batch.size)
                        }
                        biasesLayer1[i] -= learningRate * (biasesLayer1GradientAccumulator[i] / batch.size)
                    }


                }

                //now we can calculate the MSE with our predicted and actual lists of dew Points
                val mse = CalcMSE()

                //using coroutines we update our graph
                withContext(Dispatchers.Main) {
                    listener?.updateView(
                        mse = mse,
                        target = target,
                        output = output,
                        inputValues = dataset.last().first,
                        weightsLayer1 = weightsLayer1,
                        biasesLayer1 = biasesLayer1,
                        weightsLayer2 = weightsLayer2,
                        biasesLayer2 = biasesLayer2,
                        weightsOutput = Array(1) { weightsOutput },
                        biasOutput = listOf(biasOutput).toFloatArray()
                    )
                }
                println("MSE after epoch ${epoch} we have $mse")
            }
        }

    }


    // ReLU activation function
    private fun relu(x: Float) = if (x > 0) x else 0f

    private fun reluDerivative(x: Float): Float {
        return if (x > 0) 1f else 0f
    }

    fun CalcMSE(): Float {
        var MSE = 0f
        for (i in 0 until predicted.size) {
            val error = predicted[i] - actual[i]
            MSE += error * error
        }
        return MSE / (2 * predicted.size)
    }

    interface NeuralNetworkListener {
        fun updateView(
            mse: Float,
            target: Float,
            output: Float,
            inputValues: FloatArray,
            weightsLayer1: Array<FloatArray>,
            biasesLayer1: FloatArray,
            weightsLayer2: Array<FloatArray>,
            biasesLayer2: FloatArray,
            weightsOutput: Array<FloatArray>,
            biasOutput: FloatArray
        )
    }


}