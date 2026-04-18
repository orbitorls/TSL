'use client'

import { useState, useEffect } from 'react'

interface PredictionDisplayProps {
  prediction: string | null
  confidence: number
  isProcessing: boolean
}

export default function PredictionDisplay({ 
  prediction, 
  confidence,
  isProcessing 
}: PredictionDisplayProps) {
  const [displayText, setDisplayText] = useState('')

  useEffect(() => {
    if (prediction) {
      setDisplayText(prediction)
    }
  }, [prediction])

  return (
    <div className="bg-gradient-to-br from-white to-surface-50 rounded-2xl shadow-2xl p-6 border border-surface-100">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900 font-display">ผลการแปล</h2>
        <div className="text-sm text-gray-500">Realtime · TSL-51</div>
      </div>

      {/* Main prediction */}
      <div className="min-h-[140px] flex items-center justify-center rounded-lg bg-white/50 p-4">
        {isProcessing ? (
          <div className="flex flex-col items-center gap-2 text-gray-600 font-body">
            <div className="w-8 h-8 border-4 border-gray-200 border-t-brand-500 rounded-full animate-spin" />
            <div>กำลังประมวลผล...</div>
          </div>
        ) : displayText ? (
          <div className="text-6xl font-bold text-brand-600 text-center font-display">
            {displayText}
          </div>
        ) : (
          <div className="text-center">
            <p className="text-gray-400 text-lg font-body">ยกมือขึ้นหน้ากล้องเพื่อเริ่มแปล</p>
            <p className="text-sm text-gray-400 mt-2">คำแนะนำ: วางมือในกรอบ มองกล้องให้ชัด</p>
          </div>
        )}
      </div>

      {/* Confidence bar */}
      {confidence > 0 && !isProcessing && (
        <div className="mt-4">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-2 font-body">
            <span>ความมั่นใจ</span>
            <span className="font-semibold">{Math.round(confidence * 100)}%</span>
          </div>
          <div className="w-full bg-surface-200 rounded-full h-3 overflow-hidden">
            <div 
              className="bg-brand-500 h-3 rounded-full transition-all duration-300"
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
