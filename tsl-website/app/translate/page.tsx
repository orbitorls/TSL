'use client'

import { useState, useCallback, useRef } from 'react'
import Navigation from '@/components/Navigation'
import CameraView from '@/components/CameraView'
import PredictionDisplay from '@/components/PredictionDisplay'
import { predict } from '@/lib/model'

export default function TranslatePage() {
  const [prediction, setPrediction] = useState<string | null>(null)
  const [confidence, setConfidence] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [history, setHistory] = useState<string[]>([])
  
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  const handleFeatures = useCallback(async (features: number[]) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    setIsProcessing(true)

    timeoutRef.current = setTimeout(async () => {
      try {
        // Call server-side API for prediction (falls back to mock if not available)
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features })
        })

        if (!res.ok) {
          const body = await res.json().catch(() => ({}))
          console.error('Prediction API error', body)
          // fallback to local mock
          const result = predict(features)
          setPrediction(result.prediction)
          setConfidence(result.confidence)
          setHistory(prev => [result.prediction, ...prev.slice(0, 4)])
          return
        }

        const data = await res.json()

        // If API returned prediction directly (either from ONNX or fallback)
        if (data.prediction) {
          setPrediction(data.prediction)
          setConfidence(data.confidence ?? 0)
          setHistory(prev => [data.prediction, ...prev.slice(0, 4)])
        } else {
          // Unknown response shape - fallback
          const result = predict(features)
          setPrediction(result.prediction)
          setConfidence(result.confidence)
          setHistory(prev => [result.prediction, ...prev.slice(0, 4)])
        }
      } catch (err) {
        console.error('Prediction request failed', err)
        const result = predict(features)
        setPrediction(result.prediction)
        setConfidence(result.confidence)
        setHistory(prev => [result.prediction, ...prev.slice(0, 4)])
      } finally {
        setIsProcessing(false)
      }
    }, 300)
  }, [])

  return (
    <div className="min-h-screen bg-surface-50">
      <Navigation />
      
      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
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
