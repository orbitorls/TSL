'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import Navigation from '@/components/Navigation'
import CameraView from '@/components/CameraView'
import PredictionDisplay from '@/components/PredictionDisplay'
import { loadModel, predict, getModelInfo } from '@/lib/model'

export default function TranslatePage() {
  const [prediction, setPrediction] = useState<string | null>(null)
  const [confidence, setConfidence] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [history, setHistory] = useState<string[]>([])
  const [modelLoaded, setModelLoaded] = useState(false)
  const [modelInfo, setModelInfo] = useState<any>(null)

  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Load model on mount
  useEffect(() => {
    async function init() {
      const loaded = await loadModel()
      setModelLoaded(loaded)
      setModelInfo(getModelInfo())
    }
    init()
  }, [])

  const handleFeatures = useCallback(async (features: number[]) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    setIsProcessing(true)

    timeoutRef.current = setTimeout(async () => {
      try {
        // Use local model inference
        const result = await predict(features)
        setPrediction(result.prediction)
        setConfidence(result.confidence)
        setHistory(prev => [result.prediction, ...prev.slice(0, 4)])
      } catch (err) {
        console.error('Prediction error:', err)
      } finally {
        setIsProcessing(false)
      }
    }, 300)
  }, [])

  return (
    <div className="min-h-screen bg-surface-50">
      <Navigation />

      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        {/* Model Status Banner */}
        <div className="mb-6 bg-white rounded-xl shadow-sm p-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`w-3 h-3 rounded-full ${modelLoaded ? 'bg-green-500' : 'bg-yellow-500'}`} />
            <span className="text-sm text-gray-600">
              {modelLoaded ? 'Model พร้อมใช้งาน' : 'กำลังโหลด Model...'}
            </span>
            {modelInfo && (
              <span className="text-xs text-gray-400">
                v{modelInfo.config.version} | {modelInfo.config.numClasses} classes | {modelInfo.config.cvAccuracy}% CV accuracy
              </span>
            )}
          </div>
          <div className="text-xs text-gray-400">
            Real-time Inference
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Camera section */}
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-6 font-display">
              กล้อง
            </h1>
            <CameraView onPrediction={handleFeatures} />

            <p className="mt-4 text-sm text-gray-600 font-body">
              วางมือหน้ากล้องให้ชัดเจน ระยะห่างประมาณ 30-50 ซม.
            </p>
          </div>

          {/* Prediction section */}
          <div>
            <PredictionDisplay
              prediction={prediction}
              confidence={confidence}
              isProcessing={isProcessing}
            />

            {/* History */}
            {history.length > 0 && (
              <div className="mt-8 bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 font-display">
                  ประวัติการแปล
                </h3>
                <div className="space-y-2">
                  {history.map((item, index) => (
                    <div
                      key={index}
                      className="text-lg text-gray-700 border-b border-surface-100 pb-2 font-body"
                    >
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}