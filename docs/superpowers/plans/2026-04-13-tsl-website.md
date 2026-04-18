# TSL-51 Sign Language Translation Website

> **For agentic workers:** Use subagent-driven-development or executing-plans to implement task-by-task.

**Goal:** สร้างเว็บไซต์สำหรับแปลงภาษามือ (Thai Sign Language) เป็นข้อความแบบ Real-time ผ่านกล้อง

**Architecture:** Next.js frontend + API route สำหรับเรียกใช้ PyTorch model ผ่าน Python backend หรือ ONNX runtime

**Tech Stack:**
- Frontend: Next.js 14, React, TypeScript
- Styling: Tailwind CSS
- Model: PyTorch GRU (ONNX export)
- Deployment: Vercel

---

## Design Context

### Users
- คนหูหนวก / ผู้ใช้ภาษามือ ที่ต้องการ translate ภาษามือเป็นข้อความแบบเรียลไทม์
- ใช้งานผ่าน webcam/camera บนมือถือหรือคอมพิวเตอร์

### Brand Personality
- Minimal & Clean - เรียบง่าย ไม่มีสิ่งที่ไม่จำเป็น
- Accessible - เน้นการเข้าถึงได้ง่าย
- Warm & Welcoming - เป็นมิตรกับผู้ใช้

### Features
- Real-time camera translation
- รองรับ MediaPipe landmarks extraction ใน browser
- แสดงผลลัพธ์เป็น Thai text

---

## File Structure

```
tsl-website/
├── app/
│   ├── page.tsx                 # Landing page
│   ├── translate/
│   │   └── page.tsx             # Main translation page
│   ├── api/
│   │   └── predict/
│   │       └── route.ts         # Prediction API
│   └── layout.tsx               # Root layout
├── components/
│   ├── CameraView.tsx          # Camera component with MediaPipe
│   ├── PredictionDisplay.tsx    # Show predicted text
│   └── Navigation.tsx          # Navbar
├── lib/
│   ├── model.ts               # Model loading & inference
│   ├── mediapipe.ts           # MediaPipe utilities
│   └── types.ts                # TypeScript types
├── public/
│   └── models/                 # ONNX model files
├── tailwind.config.ts
├── next.config.js
└── package.json
```

---

## Tasks

### Task 1: Initialize Next.js Project

**Files:**
- Create: `tsl-website/package.json`
- Create: `tsl-website/next.config.js`
- Create: `tsl-website/tailwind.config.ts`
- Create: `tsl-website/tsconfig.json`

- [ ] **Step 1: Create project structure**

```bash
mkdir -p tsl-website
cd tsl-website
npm init -y
npm install next@14 react@18 react-dom@18 typescript @types/react @types/node tailwindcss postcss autoprefixer
npm install @mediapipe/tasks-vision
npx tailwindcss init -p
```

- [ ] **Step 2: Configure TypeScript & Next.js**

```typescript
// tsconfig.json
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

```javascript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
  },
}

module.exports = nextConfig
```

```typescript
// tailwind.config.ts
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
        }
      }
    },
  },
  plugins: [],
}
export default config
```

- [ ] **Step 3: Create basic app structure**

```typescript
// app/layout.tsx
import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'TSL-51 | Thai Sign Language Translator',
  description: 'Real-time Thai Sign Language translation using AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="th">
      <body className="min-h-screen bg-gray-50">{children}</body>
    </html>
  )
}
```

```css
// app/globals.css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
```

---

### Task 2: Create Navigation & Layout Components

**Files:**
- Create: `tsl-website/components/Navigation.tsx`
- Create: `tsl-website/app/page.tsx`

- [ ] **Step 1: Create Navigation component**

```typescript
// components/Navigation.tsx
import Link from 'next/link'

export default function Navigation() {
  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <Link href="/" className="flex-shrink-0 flex items-center">
              <span className="text-xl font-semibold text-gray-900">
                TSL-51
              </span>
            </Link>
            <div className="ml-6 flex space-x-8">
              <Link
                href="/"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900"
              >
                Home
              </Link>
              <Link
                href="/translate"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-500 hover:text-gray-900"
              >
                Translate
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
```

- [ ] **Step 2: Create Landing page**

```typescript
// app/page.tsx
import Navigation from '@/components/Navigation'
import Link from 'next/link'

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      <main className="max-w-7xl mx-auto py-16 sm:py-24 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
            Thai Sign Language
            <span className="block text-primary-600">Translation</span>
          </h1>
          <p className="mt-6 text-lg leading-8 text-gray-600 max-w-2xl mx-auto">
            แปลงภาษามือไทยเป็นข้อความแบบเรียลไทม์ด้วย AI
            ช่วยให้การสื่อสารกับผู้พิการทางการได้ยินเป็นไปได้ง่ายขึ้น
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link
              href="/translate"
              className="rounded-full bg-primary-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-primary-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-600"
            >
              เริ่มแปลภาษามือ
            </Link>
          </div>
        </div>
        
        {/* Features */}
        <div className="mt-24 grid grid-cols-1 gap-8 sm:grid-cols-3">
          <div className="text-center">
            <div className="text-3xl mb-2">📹</div>
            <h3 className="text-lg font-semibold">Real-time</h3>
            <p className="text-gray-600">แปลทันที ขณะใช้กล้อง</p>
          </div>
          <div className="text-center">
            <div className="text-3xl mb-2">🤖</div>
            <h3 className="text-lg font-semibold">AI Powered</h3>
            <p className="text-gray-600">ใช้ Deep Learning Model</p>
          </div>
          <div className="text-center">
            <div className="text-3xl mb-2">🌐</div>
            <h3 className="text-lg font-semibold">51 คำ</h3>
            <p className="text-gray-600">รองรับคำศัพท์พื้นฐาน</p>
          </div>
        </div>
      </main>
    </div>
  )
}
```

---

### Task 3: Create Camera Component with MediaPipe

**Files:**
- Create: `tsl-website/lib/mediapipe.ts`
- Create: `tsl-website/components/CameraView.tsx`

- [ ] **Step 1: Create MediaPipe utilities**

```typescript
// lib/mediapipe.ts
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'

export async function createHandLandmarker(): Promise<HandLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  )
  
  const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numHands: 2
  })
  
  return handLandmarker
}

export function extractFeatures(landmarks: any[]): number[] {
  const features: number[] = []
  
  // Left hand (63 features)
  if (landmarks[0]) {
    for (let i = 0; i < 21; i++) {
      features.push(landmarks[0][i].x, landmarks[0][i].y, landmarks[0][i].z)
    }
  } else {
    features.push(...new Array(63).fill(0))
  }
  
  // Right hand (63 features)
  if (landmarks[1]) {
    for (let i = 0; i < 21; i++) {
      features.push(landmarks[1][i].x, landmarks[1][i].y, landmarks[1][i].z)
    }
  } else {
    features.push(...new Array(63).fill(0))
  }
  
  // Pose landmarks (36 features) - simplified version
  // For now, use zeros as placeholder
  features.push(...new Array(36).fill(0))
  
  return features
}
```

- [ ] **Step 2: Create Camera component**

```typescript
// components/CameraView.tsx
'use client'

import { useEffect, useRef, useState } from 'react'
import { createHandLandmarker, extractFeatures } from '@/lib/mediapipe'

interface CameraViewProps {
  onPrediction: (features: number[]) => void
}

export default function CameraView({ onPrediction }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const landmarkerRef = useRef<any>(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let animationFrame: number
    let lastTimestamp = -1

    async function setupCamera() {
      try {
        // Create hand landmarker
        landmarkerRef.current = await createHandLandmarker()
        setIsReady(true)

        // Get camera stream
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 }
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()

          // Start detection loop
          detect()
        }
      } catch (err) {
        setError('Cannot access camera. Please allow camera permission.')
        console.error(err)
      }
    }

    function detect() {
      if (!videoRef.current || !landmarkerRef.current || !canvasRef.current) return

      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      function processFrame(timestamp: number) {
        if (timestamp - lastTimestamp < 100) { // Limit to 10fps
          animationFrame = requestAnimationFrame(processFrame)
          return
        }
        lastTimestamp = timestamp

        if (video.readyState < 2) {
          animationFrame = requestAnimationFrame(processFrame)
          return
        }

        // Draw video frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        // Detect hands
        const results = landmarkerRef.current.detectForVideo(video, timestamp)

        // Draw landmarks if detected
        if (results.landmarks && results.landmarks.length > 0) {
          drawHandLandmarks(ctx, results.landmarks)
          
          // Extract features and callback
          const features = extractFeatures(results.landmarks)
          onPrediction(features)
        }

        animationFrame = requestAnimationFrame(processFrame)
      }

      animationFrame = requestAnimationFrame(processFrame)
    }

    function drawHandLandmarks(ctx: CanvasRenderingContext2D, landmarks: any[]) {
      // Draw hand connections
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // index
        [0, 9], [9, 10], [10, 11], [11, 12], // middle
        [0, 13], [13, 14], [14, 15], [15, 16], // ring
        [0, 17], [17, 18], [18, 19], [19, 20], // pinky
        [5, 9], [9, 13], [13, 17] // palm
      ]

      ctx.fillStyle = '#0ea5e9'
      ctx.strokeStyle = '#0ea5e9'
      ctx.lineWidth = 2

      for (const hand of landmarks) {
        // Draw points
        for (const point of hand) {
          const x = point.x * canvas.width
          const y = point.y * canvas.height
          ctx.beginPath()
          ctx.arc(x, y, 4, 0, 2 * Math.PI)
          ctx.fill()
        }

        // Draw connections
        for (const [start, end] of connections) {
          const p1 = hand[start]
          const p2 = hand[end]
          ctx.beginPath()
          ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height)
          ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height)
          ctx.stroke()
        }
      }
    }

    setupCamera()

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
    }
  }, [onPrediction])

  if (error) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">
        <p className="text-red-600">{error}</p>
      </div>
    )
  }

  return (
    <div className="relative rounded-lg overflow-hidden bg-gray-900">
      <video
        ref={videoRef}
        className="w-full h-auto"
        playsInline
        muted
        style={{ display: 'none' }}
      />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="w-full h-auto"
      />
      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50">
          <p className="text-white">Loading camera...</p>
        </div>
      )}
    </div>
  )
}
```

---

### Task 4: Create Prediction Display Component

**Files:**
- Create: `tsl-website/components/PredictionDisplay.tsx`

- [ ] **Step 1: Create Prediction Display**

```typescript
// components/PredictionDisplay.tsx
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
    <div className="bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">
        ผลการแปล
      </h2>
      
      {/* Main prediction */}
      <div className="min-h-[120px] flex items-center justify-center">
        {isProcessing ? (
          <div className="flex items-center gap-2 text-gray-500">
            <div className="w-4 h-4 border-2 border-gray-300 border-t-primary-600 rounded-full animate-spin" />
            <span>กำลังประมวลผล...</span>
          </div>
        ) : displayText ? (
          <div className="text-5xl font-bold text-primary-600">
            {displayText}
          </div>
        ) : (
          <p className="text-gray-400 text-lg">
            ยกมือขึ้นหน้ากล้องเพื่อเริ่มแปล
          </p>
        )}
      </div>
      
      {/* Confidence bar */}
      {confidence > 0 && (
        <div className="mt-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>ความมั่นใจ</span>
            <span>{Math.round(confidence * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-primary-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
```

---

### Task 5: Create Translation Page

**Files:**
- Create: `tsl-website/app/translate/page.tsx`

- [ ] **Step 1: Create Translation page**

```typescript
// app/translate/page.tsx
'use client'

import { useState, useCallback, useRef } from 'react'
import Navigation from '@/components/Navigation'
import CameraView from '@/components/CameraView'
import PredictionDisplay from '@/components/PredictionDisplay'

export default function TranslatePage() {
  const [prediction, setPrediction] = useState<string | null>(null)
  const [confidence, setConfidence] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [history, setHistory] = useState<string[]>([])
  
  // Debounce prediction calls
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  const handleFeatures = useCallback((features: number[]) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    setIsProcessing(true)

    timeoutRef.current = setTimeout(async () => {
      try {
        // Send to API
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features })
        })

        const data = await response.json()
        if (data.prediction) {
          setPrediction(data.prediction)
          setConfidence(data.confidence)
          
          // Add to history
          setHistory(prev => [data.prediction, ...prev.slice(0, 4)])
        }
      } catch (err) {
        console.error('Prediction error:', err)
      } finally {
        setIsProcessing(false)
      }
    }, 300) // Debounce 300ms
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Camera section */}
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-6">
              กล้อง
            </h1>
            <CameraView onPrediction={handleFeatures} />
            
            <p className="mt-4 text-sm text-gray-600">
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
              <div className="mt-8 bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  ประวัติการแปล
                </h3>
                <div className="space-y-2">
                  {history.map((item, index) => (
                    <div 
                      key={index}
                      className="text-lg text-gray-700 border-b border-gray-100 pb-2"
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
```

---

### Task 6: Create Prediction API

**Files:**
- Create: `tsl-website/app/api/predict/route.ts`
- Create: `tsl-website/lib/model.ts`

- [ ] **Step 1: Create model utilities**

```typescript
// lib/model.ts
// This would run on server-side with ONNX runtime
// For now, we'll create a mock that returns random predictions

const CLASSES = [
  'กรุงเทพ', 'กิน', 'กลัว', 'ขอบคุณ', 'ข้าว', 'ขนมปัง',
  'คุณ', 'ฉัน', 'ชื่อ', 'ชอบ', 'ดี', 'ด้วยกัน',
  'ตลาด', 'ทำงาน', 'ทำไม', 'ที่ไหน', 'น้อง', 'น้ำ',
  'บ้าน', 'ประเทศ', 'ผู้', 'พ่อ', 'พี่', 'พรุ่งนี้',
  'ภาษามือ', 'มะม่วง', 'แม่', 'แมว', 'โรงเรียน',
  'วันนี้', 'วันหยุด', 'สวัสดี', 'สบายดี', 'หูหนวก',
  'หญิง', 'อะไร', 'อ่าน', 'อย่า', 'อยู่บ้าน',
  'เกิด', 'เรียน', 'เรียก', 'เช้า', 'เที่ยว',
  'เหงา', 'เหนื่อย', 'แต่งงาน', 'โกรธ', 'โสด', 'null_act'
]

export function predict(features: number[]): { prediction: string; confidence: number } {
  // Mock prediction - replace with actual ONNX model inference
  // In production, use ONNX Runtime for server-side inference
  
  // For demo, return random class
  const randomIndex = Math.floor(Math.random() * (CLASSES.length - 1))
  const confidence = 0.5 + Math.random() * 0.5
  
  return {
    prediction: CLASSES[randomIndex],
    confidence
  }
}
```

- [ ] **Step 2: Create API route**

```typescript
// app/api/predict/route.ts
import { NextRequest, NextResponse } from 'next/server'
import { predict } from '@/lib/model'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { features } = body

    if (!features || !Array.isArray(features)) {
      return NextResponse.json(
        { error: 'Invalid features array' },
        { status: 400 }
      )
    }

    // Run prediction
    const result = predict(features)

    return NextResponse.json(result)
  } catch (error) {
    console.error('Prediction error:', error)
    return NextResponse.json(
      { error: 'Prediction failed' },
      { status: 500 }
    )
  }
}
```

---

### Task 7: Add Model to Website

**Files:**
- Download: TSL-51 GRU model (ONNX format)
- Create: `tsl-website/public/models/tsl51_gru.onnx`

- [ ] **Step 1: Export model to ONNX**

Run on local machine with GPU:
```python
import torch
import torch.onnx

# Load trained model
model = GRUModel(input_dim=162, num_classes=51, hidden_dim=256, num_layers=3)
model.load_state_dict(torch.load('models/tsl51_gru_best.pt')['state_dict'])
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 162)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'tsl51_gru.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

- [ ] **Step 2: Upload to Vercel**

Add to `vercel.json` for serverless function:
```json
{
  "functions": {
    "api/predict/**": {
      "memory": 1024,
      "maxDuration": 60
    }
  }
}
```

---

### Task 8: Deploy to Vercel

**Files:**
- Create: `tsl-website/vercel.json`
- Modify: `package.json`

- [ ] **Step 1: Prepare for deployment**

```json
// vercel.json
{
  "framework": "nextjs",
  "functions": {
    "api/predict/**": {
      "memory": 2048,
      "maxDuration": 60
    }
  }
}
```

- [ ] **Step 2: Deploy command**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

---

## Summary

| Task | Description | Status |
|------|-------------|--------|
| 1 | Initialize Next.js Project | ⬜ |
| 2 | Navigation & Layout | ⬜ |
| 3 | Camera + MediaPipe | ⬜ |
| 4 | Prediction Display | ⬜ |
| 5 | Translation Page | ⬜ |
| 6 | Prediction API | ⬜ |
| 7 | Add Model | ⬜ |
| 8 | Deploy to Vercel | ⬜ |

**Plan complete!** Ready for execution.
