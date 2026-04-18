'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { createHandTracker, HandTracker, generatePlaceholderFeatures } from '@/lib/handtracker'

interface CameraViewProps {
  onPrediction: (features: number[]) => void
}

export default function CameraView({ onPrediction }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationRef = useRef<number>(0)
  const trackerRef = useRef<HandTracker | null>(null)
  const lastTimestampRef = useRef<number>(0)
  const useMpRef = useRef<boolean>(true)
  const isProcessingRef = useRef<boolean>(false)
  
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [useMediaPipe, setUseMediaPipe] = useState(true)

  // Handle features
  const handlePrediction = useCallback((features: number[]) => {
    if (features.length === 162) {
      onPrediction(features)
    }
  }, [onPrediction])

  useEffect(() => {
    let isMounted = true

    async function setupCamera() {
      try {
        if (streamRef.current) return

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          }
        })
        
        streamRef.current = stream
        
        if (videoRef.current && isMounted) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
          
          // Try to initialize MediaPipe
          if (useMpRef.current) {
            try {
              trackerRef.current = await createHandTracker()
              console.log('MediaPipe hand tracker initialized')
            } catch (e) {
              console.warn('MediaPipe not available, using placeholder features', e)
              useMpRef.current = false
              setUseMediaPipe(false)
            }
          }
          
          setIsReady(true)
        }
      } catch (err) {
        if (isMounted) {
          setError('ไม่สามารถเข้าถึงกล้องได้ กรุณาอนุญาติการใช้กล้อง')
          console.error('Camera error:', err)
        }
      }
    }

    async function detectFrame() {
      if (!isMounted) return
      
      const video = videoRef.current
      const canvas = canvasRef.current
      
      if (!video || !canvas || video.readyState < 2) {
        animationRef.current = requestAnimationFrame(detectFrame)
        return
      }

      const ctx = canvas.getContext('2d')
      if (!ctx) {
        animationRef.current = requestAnimationFrame(detectFrame)
        return
      }

      // Draw video to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      if (isReady && !isProcessingRef.current) {
        isProcessingRef.current = true
        
        const timestamp = video.currentTime * 1000 // Convert to milliseconds
        
        // Try MediaPipe detection if available
        if (useMpRef.current && trackerRef.current) {
          try {
            const result = await trackerRef.current.detect(video, timestamp)
            if (result && result.length === 162) {
              handlePrediction(result)
              isProcessingRef.current = false
              lastTimestampRef.current = timestamp
              animationRef.current = requestAnimationFrame(detectFrame)
              return
            }
          } catch (e) {
            console.warn('Hand detection error', e)
          }
        }
        
        // Fallback to placeholder features if MediaPipe failed
        const features = generatePlaceholderFeatures()
        handlePrediction(features)
        isProcessingRef.current = false
      }

      animationRef.current = requestAnimationFrame(detectFrame)
    }

    setupCamera().then(() => {
      if (isMounted && videoRef.current) {
        detectFrame()
      }
    })

    return () => {
      isMounted = false
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
      if (trackerRef.current) {
        trackerRef.current.close()
      }
    }
  }, [handlePrediction])

  if (error) {
    return (
      <div className="flex items-center justify-center h-96 bg-surface-100 rounded-2xl">
        <p className="text-red-600 text-center px-4 font-body">{error}</p>
      </div>
    )
  }

  return (
    <div className="relative rounded-2xl overflow-hidden bg-gradient-to-b from-gray-800 to-gray-900 shadow-xl glass-card">
      <video
        ref={videoRef}
        className="w-full h-auto hidden"
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="w-full h-auto"
      />

      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/40">
          <div className="flex flex-col items-center gap-3 text-white font-body">
            <div className="w-10 h-10 border-4 border-white/30 border-t-white rounded-full animate-spin" />
            <div className="text-lg">กำลังเชื่อมต่อกล้อง...</div>
            <div className="text-sm opacity-80">อนุญาตการเข้าถึงกล้องเมื่อเบราว์เซอร์ถาม</div>
          </div>
        </div>
      )}

      <div className="absolute top-4 left-4 bg-black/40 text-white text-xs px-3 py-1.5 rounded-full font-body text-sm flex items-center gap-2">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="opacity-90"><path d="M12 7a5 5 0 100 10 5 5 0 000-10z" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
        <span>Camera</span>
      </div>
      
      <div className="absolute top-4 right-4 text-xs text-white/70 font-body">
        {useMediaPipe ? 'MediaPipe' : 'Demo'}
      </div>
    </div>
  )
}