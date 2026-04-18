'use client'

// ============================================================
// TSL: Thai Sign Language - MediaPipe Hand Tracking
// ============================================================
// This module provides hand landmark detection using MediaPipe
// and converts to 162-dim feature vector for the model.
//
// Feature format (162): 
//   - Left hand: 21 points × 3 (x, y, z) = 63
//   - Right hand: 21 points × 3 = 63  
//   - Pose: 12 points × 3 = 36
// ============================================================

export interface HandLandmarks {
  x: number
  y: number
  z: number
}

export interface HandTracker {
  detect: (video: HTMLVideoElement, timestamp: number) => Promise<number[] | null>
  close: () => void
}

// Extract features from MediaPipe landmarks (same as training format)
export function extractFeatures(landmarks: HandLandmarks[][]): number[] {
  const features: number[] = []
  
  // Left hand (21 points × 3 = 63)
  if (landmarks[0] && landmarks[0].length >= 21) {
    for (let i = 0; i < 21; i++) {
      features.push(landmarks[0][i].x, landmarks[0][i].y, landmarks[0][i].z)
    }
  } else {
    features.push(...new Array(63).fill(0))
  }
  
  // Right hand (63)
  if (landmarks[1] && landmarks[1].length >= 21) {
    for (let i = 0; i < 21; i++) {
      features.push(landmarks[1][i].x, landmarks[1][i].y, landmarks[1][i].z)
    }
  } else {
    features.push(...new Array(63).fill(0))
  }
  
  // Pose (36) - placeholder
  features.push(...new Array(36).fill(0))
  
  return features
}

// Draw hand landmarks on canvas
export function drawHandLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: HandLandmarks[],
  width: number,
  height: number
) {
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

  // Draw points
  for (const point of landmarks) {
    const x = point.x * width
    const y = point.y * height
    ctx.beginPath()
    ctx.arc(x, y, 4, 0, 2 * Math.PI)
    ctx.fill()
  }

  // Draw connections
  for (const [start, end] of connections) {
    if (landmarks[start] && landmarks[end]) {
      ctx.beginPath()
      ctx.moveTo(landmarks[start].x * width, landmarks[start].y * height)
      ctx.lineTo(landmarks[end].x * width, landmarks[end].y * height)
      ctx.stroke()
    }
  }
}

// Create hand tracker with MediaPipe
export async function createHandTracker(): Promise<HandTracker> {
  // Lazy load MediaPipe tasks
  const vision = await import('@mediapipe/tasks-vision')
  
  // Get the FilesetResolver
  const filesetResolver = await vision.FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  )
  
  // Create HandLandmarker for 2 hands
  const handLandmarker = await vision.HandLandmarker.createFromOptions(filesetResolver, {
    numHands: 2,
    runningMode: 'VIDEO',
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/.float16/hand_landmarker.task',
      delegate: 'GPU'
    }
  })
  
  return {
    async detect(video: HTMLVideoElement, timestamp: number): Promise<number[] | null> {
      const results = handLandmarker.detectForVideo(video, timestamp)
      
      if (!results.landmarks || results.landmarks.length === 0) {
        return null
      }
      
      // Convert MediaPipe landmarks to model feature format (162-dim)
      const features = mediapipeToModelFeatures(results.landmarks)
      return features
    },
    
    close() {
      handLandmarker.close()
    }
  }
}

// Convert MediaPipe hand landmarks to model feature format (162-dim)
// Format: left_hand(63) + right_hand(63) + pose(36)
function mediapipeToModelFeatures(landmarks: any[][]): number[] {
  const features: number[] = []
  
  // MediaPipe hand landmarks: each hand has 21 points with {x, y, z}
  let leftHand: any[] | null = null
  let rightHand: any[] | null = null
  
  if (landmarks && landmarks.length > 0) {
    leftHand = landmarks[0] || null
    rightHand = landmarks[1] || null
  }
  
  // Left hand: 63 features (21 points × 3)
  if (leftHand && leftHand.length >= 21) {
    for (let i = 0; i < 21; i++) {
      features.push(leftHand[i].x ?? 0)
    }
    for (let i = 0; i < 21; i++) {
      features.push(leftHand[i].y ?? 0)
    }
    for (let i = 0; i < 21; i++) {
      features.push(leftHand[i].z ?? 0)
    }
  } else {
    features.push(...new Array(63).fill(0))
  }
  
  // Right hand: 63 features
  if (rightHand && rightHand.length >= 21) {
    for (let i = 0; i < 21; i++) {
      features.push(rightHand[i].x ?? 0)
    }
    for (let i = 0; i < 21; i++) {
      features.push(rightHand[i].y ?? 0)
    }
    for (let i = 0; i < 21; i++) {
      features.push(rightHand[i].z ?? 0)
    }
  } else {
    features.push(...new Array(63).fill(0))
  }
  
  // Pose: 36 features
  // HandLandmarker doesn't include pose - use zeros for now
  features.push(...new Array(36).fill(0))
  
  return features
}

// Fallback: generate placeholder features for demo
export function generatePlaceholderFeatures(): number[] {
  const features: number[] = []
  
  // Left hand (63)
  for (let i = 0; i < 63; i++) {
    features.push((Math.random() - 0.5) * 2)
  }
  
  // Right hand (63)
  for (let i = 0; i < 63; i++) {
    features.push((Math.random() - 0.5) * 2)
  }
  
  // Pose (36)
  for (let i = 0; i < 36; i++) {
    features.push((Math.random() - 0.5) * 2)
  }
  
  return features
}