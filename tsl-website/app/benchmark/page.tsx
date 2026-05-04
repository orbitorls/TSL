'use client'

import { useState, useEffect } from 'react'
import Navigation from '@/components/Navigation'
import { MODEL_CONFIG, MODEL_METADATA } from '@/lib/model-config'

interface BenchmarkData {
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  cvStd: number
  foldResults: number[]
  confidence: number
  perplexity: number
}

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate loading benchmark data
    const fetchData = async () => {
      // In production, this would fetch from /api/benchmark
      setData({
        accuracy: MODEL_CONFIG.cvAccuracy,
        precision: 99.86,
        recall: 99.86,
        f1Score: 99.86,
        cvStd: MODEL_CONFIG.cvStd,
        foldResults: [99.85, 99.85, 99.92, 99.84, 99.83],
        confidence: 93.30,
        perplexity: 1.07,
      })
      setLoading(false)
    }
    fetchData()
  }, [])

  const getAccuracyColor = (acc: number) => {
    if (acc >= 99) return 'text-green-600'
    if (acc >= 95) return 'text-blue-600'
    return 'text-yellow-600'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <Navigation />

      <main className="max-w-6xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-slate-900 font-display">
            Model Benchmark
          </h1>
          <p className="mt-4 text-lg text-slate-600">
            TSL-51 Thai Sign Language Recognition
          </p>
        </div>

        {loading ? (
          <div className="flex justify-center items-center py-20">
            <div className="w-12 h-12 border-4 border-slate-300 border-t-brand-500 rounded-full animate-spin" />
          </div>
        ) : data && (
          <>
            {/* Main Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
              {/* Accuracy Card */}
              <div className="bg-white rounded-2xl shadow-lg p-6 text-center">
                <div className="text-5xl font-bold text-green-600 font-display">
                  {data.accuracy.toFixed(2)}%
                </div>
                <div className="text-sm text-slate-500 mt-2">CV Accuracy</div>
                <div className="text-xs text-slate-400 mt-1">5-Fold Cross Validation</div>
              </div>

              {/* Precision Card */}
              <div className="bg-white rounded-2xl shadow-lg p-6 text-center">
                <div className="text-5xl font-bold text-blue-600 font-display">
                  {data.precision.toFixed(2)}%
                </div>
                <div className="text-sm text-slate-500 mt-2">Precision</div>
                <div className="text-xs text-slate-400 mt-1">Weighted Average</div>
              </div>

              {/* Recall Card */}
              <div className="bg-white rounded-2xl shadow-lg p-6 text-center">
                <div className="text-5xl font-bold text-purple-600 font-display">
                  {data.recall.toFixed(2)}%
                </div>
                <div className="text-sm text-slate-500 mt-2">Recall</div>
                <div className="text-xs text-slate-400 mt-1">Weighted Average</div>
              </div>

              {/* F1 Score Card */}
              <div className="bg-white rounded-2xl shadow-lg p-6 text-center">
                <div className="text-5xl font-bold text-amber-600 font-display">
                  {data.f1Score.toFixed(2)}%
                </div>
                <div className="text-sm text-slate-500 mt-2">F1 Score</div>
                <div className="text-xs text-slate-400 mt-1">Harmonic Mean</div>
              </div>
            </div>

            {/* Fold Results */}
            <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
              <h2 className="text-xl font-bold text-slate-900 mb-6 font-display">
                Cross-Validation Results
              </h2>

              <div className="flex items-end justify-between gap-4 h-48">
                {data.foldResults.map((acc, idx) => (
                  <div key={idx} className="flex flex-col items-center flex-1">
                    <div className="w-full max-w-16 bg-slate-100 rounded-lg overflow-hidden relative"
                         style={{ height: `${((acc - 99) / 1) * 100}%`, minHeight: '40px' }}>
                      <div className={`absolute inset-0 ${acc >= 99.9 ? 'bg-green-500' : 'bg-blue-500'} opacity-80`} />
                    </div>
                    <div className={`text-lg font-bold mt-2 ${getAccuracyColor(acc)}`}>
                      {acc.toFixed(2)}%
                    </div>
                    <div className="text-xs text-slate-400">Fold {idx + 1}</div>
                  </div>
                ))}
              </div>

              <div className="mt-6 flex justify-center gap-8 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-green-500 rounded" />
                  <span className="text-slate-600">Excellent (≥99.9%)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-blue-500 rounded" />
                  <span className="text-slate-600">Good (≥99%)</span>
                </div>
              </div>
            </div>

            {/* Model Info & Confidence */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Model Configuration */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-lg font-bold text-slate-900 mb-4 font-display">
                  Model Configuration
                </h2>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Version</span>
                    <span className="font-mono text-slate-900">{MODEL_CONFIG.version}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Architecture</span>
                    <span className="font-mono text-slate-900">GRU</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Classes</span>
                    <span className="font-mono text-slate-900">{MODEL_CONFIG.numClasses}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Input Dim</span>
                    <span className="font-mono text-slate-900">{MODEL_CONFIG.inputDim}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Hidden Dim</span>
                    <span className="font-mono text-slate-900">{MODEL_CONFIG.hiddenDim}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Layers</span>
                    <span className="font-mono text-slate-900">{MODEL_CONFIG.numLayers}</span>
                  </div>
                </div>
              </div>

              {/* Inference Metrics */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-lg font-bold text-slate-900 mb-4 font-display">
                  Inference Metrics
                </h2>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-slate-600">Mean Confidence</span>
                      <span className="font-mono text-slate-900">{(data.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2">
                      <div
                        className="bg-brand-500 h-2 rounded-full"
                        style={{ width: `${data.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-slate-600">Perplexity</span>
                      <span className="font-mono text-slate-900">{data.perplexity.toFixed(3)}</span>
                    </div>
                    <div className="text-xs text-slate-400">Lower is better (1.0 = perfect)</div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-slate-600">CV Std Dev</span>
                      <span className="font-mono text-slate-900">{(data.cvStd * 100).toFixed(4)}%</span>
                    </div>
                    <div className="text-xs text-slate-400">Lower = more consistent</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Dataset Info */}
            <div className="mt-8 bg-gradient-to-r from-brand-500 to-brand-600 rounded-2xl shadow-lg p-8 text-white">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
                <div>
                  <div className="text-3xl font-bold">{MODEL_METADATA.totalSamples.toLocaleString()}</div>
                  <div className="text-sm opacity-80 mt-1">Total Samples</div>
                </div>
                <div>
                  <div className="text-3xl font-bold">{MODEL_METADATA.folds}</div>
                  <div className="text-sm opacity-80 mt-1">CV Folds</div>
                </div>
                <div>
                  <div className="text-3xl font-bold">{MODEL_METADATA.augmentation}</div>
                  <div className="text-sm opacity-80 mt-1">Augmentation</div>
                </div>
                <div>
                  <div className="text-3xl font-bold">{MODEL_METADATA.trainingDate}</div>
                  <div className="text-sm opacity-80 mt-1">Training Date</div>
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}