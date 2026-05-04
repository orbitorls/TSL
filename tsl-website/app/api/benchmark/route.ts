import { NextResponse } from 'next/server'
import { MODEL_CONFIG, MODEL_METADATA } from '@/lib/model-config'

export const runtime = 'nodejs'

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    model: {
      version: MODEL_CONFIG.version,
      architecture: MODEL_CONFIG.modelType,
      numClasses: MODEL_CONFIG.numClasses,
      inputDim: MODEL_CONFIG.inputDim,
      hiddenDim: MODEL_CONFIG.hiddenDim,
      numLayers: MODEL_CONFIG.numLayers,
    },
    metrics: {
      cvAccuracy: MODEL_CONFIG.cvAccuracy,
      cvStd: MODEL_CONFIG.cvStd,
      precision: 99.86,
      recall: 99.86,
      f1Score: 99.86,
    },
    foldResults: [99.85, 99.85, 99.92, 99.84, 99.83],
    inference: {
      meanConfidence: 0.933,
      perplexity: 1.071,
    },
    metadata: MODEL_METADATA,
  })
}