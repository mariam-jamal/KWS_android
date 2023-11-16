/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.audio

import android.content.Context
import android.media.AudioRecord
import android.os.SystemClock
import android.util.Log
import com.jlibrosa.audio.JLibrosa
import org.tensorflow.lite.examples.audio.fragments.AudioClassificationListener
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import org.tensorflow.lite.task.core.BaseOptions
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit

class AudioClassificationHelper(
  val context: Context,
  val listener: AudioClassificationListener,
  var currentModel: String = YAMNET_MODEL,
  var classificationThreshold: Float = DISPLAY_THRESHOLD,
  var overlap: Float = DEFAULT_OVERLAP_VALUE,
  var numOfResults: Int = DEFAULT_NUM_OF_RESULTS,
  var currentDelegate: Int = 0,
  var numThreads: Int = 2
) {
    private lateinit var classifier: AudioClassifier
    private lateinit var tensorAudio: TensorAudio
    private lateinit var recorder: AudioRecord
    private lateinit var executor: ScheduledThreadPoolExecutor

    private val classifyRunnable = Runnable {
        classifyAudio()
    }

    init {
        initClassifier()
    }

    fun initClassifier() {
        // Set general detection options, e.g. number of used threads
        val baseOptionsBuilder = BaseOptions.builder()
            .setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU.
        // Possible to also use a GPU delegate, but this requires that the classifier be created
        // on the same thread that is using the classifier, which is outside of the scope of this
        // sample's design.
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        // Configures a set of parameters for the classifier and what results will be returned.
        val options = AudioClassifier.AudioClassifierOptions.builder()
            .setScoreThreshold(classificationThreshold)
            .setMaxResults(numOfResults)
            .setBaseOptions(baseOptionsBuilder.build())
            .build()

        try {
            // Create the classifier and required supporting objects
            classifier = AudioClassifier.createFromFileAndOptions(context, currentModel, options)
            tensorAudio = classifier.createInputTensorAudio()
            recorder = classifier.createAudioRecord()
            startAudioClassification()
        } catch (e: IllegalStateException) {
            listener.onError(
                "Audio Classifier failed to initialize. See error logs for details"
            )

            Log.e("AudioClassification", "TFLite failed to load with error: " + e.message)
        }
    }

    fun startAudioClassification() {
        if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
            return
        }

        recorder.startRecording()
        executor = ScheduledThreadPoolExecutor(1)

        // Each model will expect a specific audio recording length. This formula calculates that
        // length using the input buffer size and tensor format sample rate.
        // For example, YAMNET expects 0.975 second length recordings.
        // This needs to be in milliseconds to avoid the required Long value dropping decimals.
        val lengthInMilliSeconds = ((classifier.requiredInputBufferSize * 1.0f) /
                classifier.requiredTensorAudioFormat.sampleRate) * 1000

        val interval = (lengthInMilliSeconds * (1 - overlap)).toLong()

        executor.scheduleAtFixedRate(
            classifyRunnable,
            0,
            interval,
            TimeUnit.MILLISECONDS)
    }

    private fun preprocessAudio(waveform: FloatArray): FloatArray {
        val sampleRate = 8000 // Sample rate
        val sampleTime = 2.0
        val nfft = 512 // Number of FFT points
        val nhop = 400 // Number of audio samples between FFTs
        val cutoffFreq = 4000
        val numAvgBins = 8 // Number of bins to average together in FFT

        val paddedWaveform = waveform.copyOf(16000)

        val stftNSlices = (kotlin.math.ceil(((sampleTime * sampleRate) / nhop) -
                (nfft / nhop)) + 1).toInt()
        val stftMaxBin = ((nfft / 2) / ((sampleRate / 2) / cutoffFreq)).toInt()

        val stft = Array(stftNSlices) { FloatArray((stftMaxBin) / numAvgBins) }

        for (i in 0..stftNSlices) {
            val winStart = i * nhop
            val winStop = (i * nhop) + nfft

            var window = paddedWaveform.copyOfRange(winStart, winStop)
            if (window.size < nfft) {
                val padding = FloatArray(nfft - window.size)
                window += padding
            }

            // Apply Hanning window
            for (j in window.indices) {
                window[j] = (window[j] * 0.5 * (1 - kotlin.math.cos(2 * Math.PI * j / (nfft - 1)))).toInt()
                    .toFloat()
            }

            // Compute FFT
            val fft = FloatArray(nfft / 2)
            for (j in 0 until nfft / 2) {
                var sum = 0.0
                for (k in 0 until nfft) {
                    sum += window[k] * kotlin.math.cos(2 * Math.PI * j * k / nfft)
                }
                fft[j] = kotlin.math.sqrt(sum.toFloat() * 2).toFloat() // magnitude
            }

            // Only keep the frequency bins we care about
            val filteredFFT = fft.sliceArray(1 until stftMaxBin)

            // Average every numAvgBins bins together to reduce the size of FFT
            for (j in stft[i].indices) {
                stft[i][j] = (filteredFFT.sliceArray(j * numAvgBins until 255.coerceAtMost((j + 1) * numAvgBins))
                    .average() / nfft).toFloat() // Normalize
            }
        }

        // Flatten the 2D array to 1D array using flatMap
        val flattenedStft = stft.flatMap { it.asIterable() }.toFloatArray()

        return flattenedStft
    }

    // Inside your classifyAudio method
    private fun classifyAudio() {
        try {
            tensorAudio.load(recorder)
            val jLibrosa = JLibrosa()
            val mfccValues = jLibrosa.generateMFCCFeatures(tensorAudio.tensorBuffer.floatArray, 44100, 10)

            /*// Preprocess the audio waveform to get the feature vector
            val featureVector = preprocessAudio(tensorAudio.tensorBuffer.floatArray)

            // Create a TensorBuffer object from the feature vector
            val tensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, featureVector.size), DataType.FLOAT32)
            tensorBuffer.loadArray(featureVector, intArrayOf(featureVector.size))

            // Create a TensorAudio object with the correct shape and sample rate
            val tensorAudio = classifier.createInputTensorAudio()
            tensorAudio.load(tensorBuffer.floatArray)*/


            // Perform inference using the feature vector
            var inferenceTime = SystemClock.uptimeMillis()
            val output = classifier.classify(tensorAudio)
            inferenceTime = SystemClock.uptimeMillis() - inferenceTime
            listener.onResult(output[0].categories, inferenceTime)
        } catch (e: Exception){
            e.printStackTrace()
        }


    }

    fun stopAudioClassification() {
        recorder.stop()
        executor.shutdownNow()
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_NNAPI = 1
        const val DISPLAY_THRESHOLD = 0.3f
        const val DEFAULT_NUM_OF_RESULTS = 2
        const val DEFAULT_OVERLAP_VALUE = 0.5f
        const val YAMNET_MODEL = "model_f1score_float16_meta.tflite"
        const val SPEECH_COMMAND_MODEL = "speech.tflite"
    }
}
