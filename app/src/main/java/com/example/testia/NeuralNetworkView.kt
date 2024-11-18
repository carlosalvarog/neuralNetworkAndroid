package com.example.testia

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.View
import kotlin.math.min
class NeuralNetworkView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    private val paint = Paint()
    private val circlePaint = Paint()

    // Network structure
    private val inputNodes = 3
    private val hiddenLayer1Nodes = 3
    private val hiddenLayer2Nodes = 2
    private val outputNodes = 1

    // Dynamic values
    var inputValues = FloatArray(inputNodes) { 0f }
    var weightsLayer1 = Array(hiddenLayer1Nodes) { FloatArray(inputNodes) { 0f } }
    var biasesLayer1 = FloatArray(hiddenLayer1Nodes) { 0f }
    var weightsLayer2 = Array(hiddenLayer2Nodes) { FloatArray(hiddenLayer1Nodes) { 0f } }
    var biasesLayer2 = FloatArray(hiddenLayer2Nodes) { 0f }
    var weightsOutput = FloatArray(hiddenLayer2Nodes) { 0f } // Changed to 1D
    var biasOutput = 0f
    var outputValue = 0f
    var targetValue = 0f
    var lastMSE = 0f

    init {
        paint.color = Color.WHITE
        paint.isAntiAlias = true
        paint.style = Paint.Style.FILL

        circlePaint.color = Color.WHITE
        circlePaint.isAntiAlias = true
        circlePaint.style = Paint.Style.STROKE
    }

    override fun onDraw(canvas: Canvas) {
        try {
            super.onDraw(canvas)
            canvas.drawColor(Color.BLACK)

            val width = width.toFloat()
            val height = height.toFloat()
            val minDimension = min(width, height)

            paint.textSize = minDimension * 0.03f
            circlePaint.strokeWidth = minDimension * 0.002f

            val nodeRadius = minDimension * 0.03f
            val verticalOffset = -height * 0.1f

            // Draw layers
            drawLayer(canvas, inputNodes, width * 0.1f, height, inputValues, nodeRadius, "Input", null, verticalOffset)
            drawLayer(canvas, hiddenLayer1Nodes, width * 0.4f, height, null, nodeRadius, "Hidden1", biasesLayer1, verticalOffset)
            drawLayer(canvas, hiddenLayer2Nodes, width * 0.7f, height, null, nodeRadius, "Hidden2", biasesLayer2, verticalOffset)
            drawLayer(canvas, outputNodes, width * 0.9f, height, null, nodeRadius, "Output", floatArrayOf(biasOutput), verticalOffset)

            // Draw connections
            drawConnections(canvas, inputNodes, hiddenLayer1Nodes, width * 0.1f, width * 0.4f, height, weightsLayer1, "Input-Hidden1", verticalOffset)
            drawConnections(canvas, hiddenLayer1Nodes, hiddenLayer2Nodes, width * 0.4f, width * 0.7f, height, weightsLayer2, "Hidden1-Hidden2", verticalOffset)
            drawOutputConnections(canvas, hiddenLayer2Nodes, outputNodes, width * 0.7f, width * 0.9f, height, weightsOutput, "Hidden2-Output", verticalOffset)

            // Draw output value
            paint.textAlign = Paint.Align.LEFT
            canvas.drawText(String.format("%.2f", outputValue), width * 0.95f, height * 0.5f + verticalOffset, paint)

            // Draw MSE, target, and output values at the bottom
            drawMetrics(canvas, width, height)
        } catch (e: Exception) {
            Log.e("NeuralNetworkView", "Error in onDraw", e)
            e.printStackTrace()
        }
    }
    private fun drawLayer(
        canvas: Canvas,
        nodes: Int,
        x: Float,
        height: Float,
        values: FloatArray?,
        nodeRadius: Float,
        layerName: String,
        layerBiases: FloatArray?,
        verticalOffset: Float
    ) {
        try {
            val spacing = height / (nodes + 1)
            for (i in 0 until nodes) {
                val y = spacing * (i + 1) + verticalOffset

                // Draw neuron circle
                canvas.drawCircle(x, y, nodeRadius, circlePaint)

                // Draw value outside the circle (if applicable)
                paint.textAlign = Paint.Align.RIGHT
                val value = values?.getOrNull(i)
                if (value != null) {
                    canvas.drawText(String.format("%.2f", value), x - nodeRadius - 10f, y + paint.textSize / 3, paint)
                }

                // Draw bias inside the circle
                paint.textAlign = Paint.Align.CENTER
                val bias = layerBiases?.getOrNull(i)
                if (bias != null) {
                    canvas.drawText(String.format("%.2f", bias), x, y + paint.textSize / 3, paint)
                }
            }
        } catch (e: Exception) {
            Log.e("NeuralNetworkView", "Error in drawLayer: $layerName", e)
            e.printStackTrace()
        }
    }
    private fun drawConnections(
        canvas: Canvas,
        fromNodes: Int,
        toNodes: Int,
        fromX: Float,
        toX: Float,
        height: Float,
        weights: Array<FloatArray>,
        connectionName: String,
        verticalOffset: Float
    ) {
        try {
            val fromSpacing = height / (fromNodes + 1)
            val toSpacing = height / (toNodes + 1)
            for (i in 0 until fromNodes) {
                for (j in 0 until toNodes) {
                    val fromY = fromSpacing * (i + 1) + verticalOffset
                    val toY = toSpacing * (j + 1) + verticalOffset
                    canvas.drawLine(fromX, fromY, toX, toY, paint)
                    val weightX = fromX + (toX - fromX) * 0.2f
                    val weightY = fromY + (toY - fromY) * 0.2f
                    paint.textAlign = Paint.Align.CENTER
                    val weight = weights.getOrNull(j)?.getOrNull(i) ?: 0f
                    canvas.drawText(String.format("%.2f", weight), weightX, weightY, paint)
                }
            }
        } catch (e: Exception) {
            Log.e("NeuralNetworkView", "Error in drawConnections: $connectionName", e)
            e.printStackTrace()
        }
    }

    private fun drawMetrics(canvas: Canvas, width: Float, height: Float) {
        val baseY = height - 100f
        paint.textAlign = Paint.Align.LEFT
        canvas.drawText("MSE: ${String.format("%.6e", lastMSE)}", 50f, baseY, paint)
        canvas.drawText(" Last dew Point: ${String.format("%.2f", targetValue)}", 50f, baseY + 40f, paint)
        canvas.drawText("Last output: ${String.format("%.2f", outputValue)}", 50f, baseY + 80f, paint)
    }

    private fun drawOutputConnections(
        canvas: Canvas,
        fromNodes: Int,
        toNodes: Int,
        fromX: Float,
        toX: Float,
        height: Float,
        weights: FloatArray,
        connectionName: String,
        verticalOffset: Float
    ) {
        try {
            val fromSpacing = height / (fromNodes + 1)
            val toSpacing = height / (toNodes + 1)
            for (j in 0 until toNodes) {
                val toY = toSpacing * (j + 1) + verticalOffset
                for (i in 0 until fromNodes) {
                    val fromY = fromSpacing * (i + 1) + verticalOffset
                    val weight = weights.getOrNull(i) ?: 0f
                    canvas.drawLine(fromX, fromY, toX, toY, paint)
                    val weightX = fromX + (toX - fromX) * 0.2f
                    val weightY = fromY + (toY - fromY) * 0.2f
                    paint.textAlign = Paint.Align.CENTER
                    canvas.drawText(String.format("%.2f", weight), weightX, weightY, paint)
                }
            }
        } catch (e: Exception) {
            Log.e("NeuralNetworkView", "Error in drawOutputConnections: $connectionName", e)
            e.printStackTrace()
        }
    }

    fun updateView(
        newInputValues: FloatArray,
        newWeightsLayer1: Array<FloatArray>,
        newBiasesLayer1: FloatArray,
        newWeightsLayer2: Array<FloatArray>,
        newBiasesLayer2: FloatArray,
        newWeightsOutput: FloatArray, // Adjusted for 1D
        newBiasOutput: Float,        // Adjusted for single bias
        newOutputValue: Float,
        newTargetValue: Float,
        newMSE: Float
    ) {
        inputValues = newInputValues
        weightsLayer1 = newWeightsLayer1
        biasesLayer1 = newBiasesLayer1
        weightsLayer2 = newWeightsLayer2
        biasesLayer2 = newBiasesLayer2
        weightsOutput = newWeightsOutput
        biasOutput = newBiasOutput
        outputValue = newOutputValue
        targetValue = newTargetValue
        lastMSE = newMSE
        invalidate()
    }
}
