import { NextRequest, NextResponse } from 'next/server'
import { validateFeatures, getModelInfo, predict } from '@/lib/server-model'

export const runtime = 'nodejs'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { features } = body

    if (!features || !Array.isArray(features)) {
      return NextResponse.json({ error: 'Invalid features array' }, { status: 400 })
    }

    if (!validateFeatures(features)) {
      return NextResponse.json({ error: 'Invalid feature values' }, { status: 400 })
    }

    const result = predict(features)
    const info = getModelInfo()

    return NextResponse.json({
      ...result,
      modelVersion: info.config.version,
      fallback: true,
    })
  } catch (error) {
    console.error('Prediction error:', error)
    return NextResponse.json({ error: 'Prediction failed' }, { status: 500 })
  }
}

export async function GET() {
  const info = getModelInfo()
  return NextResponse.json({
    status: 'ok',
    model: {
      version: info.config.version,
      classes: info.config.numClasses,
      accuracy: info.config.cvAccuracy,
      loaded: true,
    },
    metadata: info.metadata,
  })
}