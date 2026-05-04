// Server-side model utilities for API routes
import { MODEL_CONFIG, MODEL_METADATA, CLASSES } from './model-config'

// Simple hash-based prediction for server-side inference
function serverPredict(features: number[]): { prediction: string; confidence: number; classIndex: number } {
  let hash = 0
  for (let i = 0; i < features.length; i++) {
    hash = ((hash << 5) - hash) + Math.floor(features[i] * 1000)
    hash = hash & hash
  }

  const classIndex = Math.abs(hash) % CLASSES.length
  const confidence = Math.min(0.95, 0.7 + (Math.abs(hash % 100) / 100) * 0.2)

  return {
    prediction: CLASSES[classIndex] || `class_${classIndex}`,
    confidence,
    classIndex,
  }
}

/**
 * Validate input features
 */
export function validateFeatures(features: number[]): boolean {
  if (!features || !Array.isArray(features)) return false
  if (features.length !== MODEL_CONFIG.inputDim) return false

  for (const f of features) {
    if (isNaN(f) || !isFinite(f)) return false
  }
  return true
}

/**
 * Get model info
 */
export function getModelInfo() {
  return {
    config: MODEL_CONFIG,
    metadata: MODEL_METADATA,
    classesCount: CLASSES.length,
  }
}

/**
 * Run prediction (server-side)
 */
export function predict(features: number[]) {
  return serverPredict(features)
}