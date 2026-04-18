import { NextRequest, NextResponse } from 'next/server'
import { predict, validateFeatures, CLASSES } from '@/lib/model'

// Note: ONNX inference requires onnxruntime-node to be installed in the deployment.
// For now, we use mock prediction. To enable ONNX:
// 1. Add onnxruntime-node to package.json dependencies
// 2. Ensure the ONNX model file is accessible
// 3. Redeploy with native module support

export const runtime = 'nodejs'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { features } = body

    if (!features || !Array.isArray(features)) {
      return NextResponse.json({ error: 'Invalid features array' }, { status: 400 })
    }

    // Validate features
    if (!validateFeatures(features)) {
      return NextResponse.json({ error: 'Invalid feature values' }, { status: 400 })
    }

    // For now, always use mock prediction
    // TODO: Add ONNX inference when onnxruntime-node is available in deployment
    const result = predict(features)
    return NextResponse.json({ ...result, fallback: true })
  } catch (error) {
    console.error('Prediction error:', error)
    return NextResponse.json({ error: 'Prediction failed' }, { status: 500 })
  }
}