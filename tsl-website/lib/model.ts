// Model prediction utilities
// In production, this would use ONNX Runtime for server-side inference

export const CLASSES = [
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

export interface PredictionResult {
  prediction: string
  confidence: number
}

// Mock prediction - replace with actual ONNX model inference in production
export function predict(features: number[]): PredictionResult {
  // Basic validation
  if (!validateFeatures(features)) {
    return { prediction: 'ไม่แน่ใจ', confidence: 0 }
  }

  // For demo purposes, return a random prediction.
  // Use full CLASSES length (previously used CLASSES.length - 1 which excluded the last class).
  const randomIndex = Math.floor(Math.random() * CLASSES.length)
  const confidence = 0.5 + Math.random() * 0.5

  return {
    prediction: CLASSES[randomIndex],
    confidence
  }
}

// Check if features are valid
export function validateFeatures(features: number[]): boolean {
  if (!features || !Array.isArray(features)) return false
  if (features.length !== 162) return false
  
  // Check for NaN or Infinity
  for (const f of features) {
    if (isNaN(f) || !isFinite(f)) return false
  }
  
  return true
}
