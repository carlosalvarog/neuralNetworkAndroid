package com.example.testia

import android.os.Bundle
import android.view.Window
import android.view.WindowManager
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : AppCompatActivity(), NeuralNetwork.NeuralNetworkListener {

    lateinit var neuralNetwork: NeuralNetwork
    private lateinit var neuralNetworkView: NeuralNetworkView


    var dewPointMin = Float.MAX_VALUE
    var dewPointMax = Float.MIN_VALUE


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        );

        // Set the content view to the custom neural network view
        setContentView(R.layout.activity_main)
        neuralNetworkView = findViewById(R.id.neuralNetworkView)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        neuralNetwork = NeuralNetwork(this)

        val samples = generateSamples(100000)
        println("Generated ${samples.size} samples:")

        neuralNetwork.train(samples, 140, 25)


    }


    fun calculateDewPoint(T: Float, RH: Float, P: Float): Float {
        val a = 17.27f
        val b = 237.7f
        val P0 = 1013.25f // Standard pressure in hPa

        // Saturation vapor pressure
        val es = 6.112f * Math.exp(((a * T) / (b + T)).toDouble()).toFloat()

        // Adjusted actual vapor pressure
        val e = (RH * P * es) / (100 * P0)

        // Dew point calculation
        return (b * Math.log(e.toDouble()).toFloat()) / (a - Math.log(e.toDouble()).toFloat())
    }

    fun normalizeTemp(temp: Float): Float {
        val tempMin = -10f
        val tempMax = 40f
        return normalize(temp, tempMin, tempMax)
    }

    fun normalizeRh(rh: Float): Float {
        val rhMin = 15f
        val rhMax = 100f
        return normalize(rh, rhMin, rhMax)
    }

    fun normalizePressure(pressure: Float): Float {
        val pressureMin = 900f
        val pressureMax = 1100f
        return normalize(pressure, pressureMin, pressureMax)
    }

    fun normalizeDewPoint(dewPoint: Float): Float {
        return normalize(dewPoint, dewPointMin, dewPointMax)
    }


    fun normalize(value: Float, min: Float, max: Float): Float {
        return (value - min) / (max - min)
    }

    fun generateSamples(numSamples: Int): List<Pair<FloatArray, Float>> {
        // Temporarily store raw (T, RH, P) and Td
        val samples = mutableListOf<Triple<Float, Float, Float>>()


        // Step 1: Generate raw samples and find the dew point range
        repeat(numSamples) {
            // Randomly sample T, RH, and P within their ranges
            val T = -10f + Math.random().toFloat() * (40f - (-10f))
            val RH = 15f + Math.random().toFloat() * (100f - 15f)
            val P = 900f + Math.random().toFloat() * (1100f - 900f)

            // Calculate dew point
            val Td = calculateDewPoint(T, RH, P)

            // Track min and max dew point
            if (Td > dewPointMax) dewPointMax = Td
            if (Td < dewPointMin) dewPointMin = Td

            // Store raw data
            samples.add(Triple(T, RH, P))
        }

        // Step 2: Normalize all values using determined ranges
        return samples.map { (T, RH, P) ->
            // Normalize T, RH, P, and Td
            val TNorm = normalizeTemp(T)
            val RHNorm = normalizeRh(RH)
            val PNorm = normalizePressure(P)
            val TdNorm = normalizeDewPoint(calculateDewPoint(T, RH, P))

            Pair(floatArrayOf(TNorm, RHNorm, PNorm), TdNorm)
        }

    }


    override fun updateView(
        mse: Float,
        target: Float,
        output: Float,
        inputValues: FloatArray,
        hiddenLayer1Weights: Array<FloatArray>,
        hiddenLayer1Biases: FloatArray,
        hiddenLayer2Weights: Array<FloatArray>,
        hiddenLayer2Biases: FloatArray,
        outputWeights: Array<FloatArray>,
        outputBiases: FloatArray
    ) {


        var target1 = target * (dewPointMax - dewPointMin) + dewPointMin
        var output1 = output * (dewPointMax - dewPointMin) + dewPointMin

        neuralNetworkView.updateView(inputValues, hiddenLayer1Weights, hiddenLayer1Biases, hiddenLayer2Weights, hiddenLayer2Biases, outputWeights[0], outputBiases[0], output1, target1, mse)

    }


}