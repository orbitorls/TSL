// TSL-51 Model Configuration
// Latest model: tsl51_gru_20260503_183943.onnx
// 262 classes | 99.86% CV accuracy | 162 features

export const MODEL_CONFIG = {
  version: '2.1.0',
  modelPath: '/models/tsl51_gru_20260503_183943.onnx',
  inputDim: 162,
  hiddenDim: 256,
  numLayers: 3,
  numClasses: 262,
  modelType: 'gru',
  cvAccuracy: 99.86,
  cvStd: 0.0319,
}

// Model metadata
export const MODEL_METADATA = {
  trainingDate: '2026-05-03',
  dataset: 'tsl51_expert_full',
  totalSamples: 164106,
  folds: 5,
  augmentation: '2x',
  labelSmoothing: 0.1,
  mixupAlpha: 0.2,
}

export interface PredictionResult {
  prediction: string
  confidence: number
  classIndex: number
  probabilities: number[]
  topK: Array<{ classIndex: number; className: string; probability: number }>
}

// Class labels (262 classes from tsl51_expert dataset)
export const CLASSES: string[] = [
  // Null/action classes
  'null_action', 'kpp_null', 'null_act',
  // Thai consonants (ก-ฮ)
  'kpp_ก', 'kpp_ข', 'kpp_ค', 'kpp_ต', 'kpp_ง', 'kpp_จ', 'kpp_ฉ', 'kpp_ช', 'kpp_ซ', 'kpp_ญ',
  'kpp_ฑ', 'kpp_ฒ', 'kpp_ณ', 'kpp_ด', 'kpp_ถ', 'kpp_ท', 'kpp_ธ', 'kpp_น', 'kpp_บ', 'kpp_ป',
  'kpp_ผ', 'kpp_ฝ', 'kpp_พ', 'kpp_ฟ', 'kpp_ภ', 'kpp_ม', 'kpp_ย', 'kpp_ร', 'kpp_ล', 'kpp_ว',
  'kpp_ศ', 'kpp_ษ', 'kpp_ส', 'kpp_ห', 'kpp_ฬ', 'kpp_อ', 'kpp_ฮ',
  // Numbers 0-9
  'kpp_0', 'kpp_1', 'kpp_2', 'kpp_3', 'kpp_4', 'kpp_5', 'kpp_6', 'kpp_7', 'kpp_8', 'kpp_9',
  // Common words
  'kpp_กรุงเทพ', 'kpp_กิน', 'kpp_กลัว', 'kpp_ขอบคุณ', 'kpp_ข้าว', 'kpp_ขนมปัง',
  'kpp_คุณ', 'kpp_ฉัน', 'kpp_ชื่อ', 'kpp_ชอบ', 'kpp_ดี', 'kpp_ด้วยกัน',
  'kpp_ตลาด', 'kpp_ทำงาน', 'kpp_ทำไม', 'kpp_ที่ไหน', 'kpp_น้อง', 'kpp_น้ำ',
  'kpp_บ้าน', 'kpp_ประเทศ', 'kpp_ผู้', 'kpp_พ่อ', 'kpp_พี่', 'kpp_พรุ่งนี้',
  'kpp_ภาษามือ', 'kpp_มะม่วง', 'kpp_แม่', 'kpp_แมว', 'kpp_โรงเรียน',
  'kpp_วันนี้', 'kpp_วันหยุด', 'kpp_สวัสดี', 'kpp_สบายดี', 'kpp_หูหนวก',
  'kpp_หญิง', 'kpp_อะไร', 'kpp_อ่าน', 'kpp_อย่า', 'kpp_อยู่บ้าน',
  'kpp_เกิด', 'kpp_เรียน', 'kpp_เรียก', 'kpp_เช้า', 'kpp_เที่ยว',
  'kpp_เหงา', 'kpp_เหนื่อย', 'kpp_แต่งงาน', 'kpp_โกรธ', 'kpp_โสด',
  // Extended vocabulary
  'kpp_สิ่ง', 'kpp_รถ', 'kpp_ไฟ', 'kpp_นาที', 'kpp_ชั่วโมง', 'kpp_เดือน', 'kpp_ปี',
  'kpp_คน', 'kpp_หญิง', 'kpp_ชาย', 'kpp_เด็ก', 'kpp_ครอบครัว', 'kpp_แม่',
  'kpp_พ่อ', 'kpp_ลูก', 'kpp_ปู่', 'kpp_ย่า', 'kpp_ตา', 'kpp_ยาย',
  'kpp_พี่ชาย', 'kpp_พี่สาว', 'kpp_น้องชาย', 'kpp_น้องสาว', 'kpp_ลุง', 'kpp_ป้า',
  'kpp_น้า', 'kpp_อา', 'kpp_ปลา', 'kpp_หมู', 'kpp_ไก่', 'kpp_กุ้ง', 'kpp_ปู',
  'kpp_ผัก', 'kpp_ผลไม้', 'kpp_ส้ม', 'kpp_กล้วย', 'kpp_มังคุด', 'kpp_ทุเรียน',
  'kpp_ข้าวเหนียว', 'kpp_ข้าวผัด', 'kpp_ผัดไทย', 'kpp_ต้มยำ', 'kpp_แกงเขียว',
  'kpp_ส้า', 'kpp_ลาบ', 'kpp_น้ำพริก', 'kpp_ข้าวตัง', 'kpp_ไข่', 'kpp_นม',
  'kpp_กาแฟ', 'kpp_ชา', 'kpp_น้ำส้ม', 'kpp_โค้ก', 'kpp_เบียร์', 'kpp_ไวน์',
  // Places
  'kpp_กรุงเทพฯ', 'kpp_เชียงใหม่', 'kpp_ภูเก็ต', 'kpp_ขอนแก่น', 'kpp_สงขลา',
  'kpp_โรงพยาบาล', 'kpp_สถานีตำรวจ', 'kpp_โรงเรียน', 'kpp_มหาวิทยาลัย', 'kpp_วัด',
  'kpp_ตลาด', 'kpp_ห้าง', 'kpp_ธนาคาร', 'kpp_ไปรษณีย์', 'kpp_สถานีรถไฟ',
  // Time expressions
  'kpp_วันจันทร์', 'kpp_วันอังคาร', 'kpp_วันพุธ', 'kpp_วันพฤหัส', 'kpp_วันศุกร์',
  'kpp_วันเสาร์', 'kpp_วันอาทิตย์', 'kpp_เช้า', 'kpp_เที่ยง', 'kpp_บ่าย', 'kpp_เย็น', 'kpp_กลางคืน',
  // Actions
  'kpp_เดิน', 'kpp_วิ่ง', 'kpp_นั่ง', 'kpp_ยืน', 'kpp_นอน', 'kpp_กิน', 'kpp_ดื่ม',
  'kpp_พูด', 'kpp_ฟัง', 'kpp_มอง', 'kpp_เขียน', 'kpp_อ่าน', 'kpp_เรียน', 'kpp_ทำงาน',
  'kpp_เล่น', 'kpp_นอนหลับ', 'kpp_ตื่น', 'kpp_อาบน้ำ', 'kpp_แต่งตัว', 'kpp_กินข้าว',
  // Emotions
  'kpp_ดีใจ', 'kpp_เสียใจ', 'kpp_โกรธ', 'kpp_กลัว', 'kpp_เหงา', 'kpp_รัก', 'kpp_เกลียด',
  // More vocabulary
  'kpp_ใหญ่', 'kpp_เล็ก', 'kpp_สูง', 'kpp_ต่ำ', 'kpp_ยาว', 'kpp_สั้น', 'kpp_หนัก', 'kpp_เบา',
  'kpp_ใหม่', 'kpp_เก่า', 'kpp_ดี', 'kpp_เลว', 'kpp_งาม', 'kpp_ขี้เหร่',
  'kpp_ร้อน', 'kpp_เย็น', 'kpp_อุ่น', 'kpp_หวาน', 'kpp_เค็ม', 'kpp_เปรี้ยว', 'kpp_ขม',
  'kpp_เปิด', 'kpp_ปิด', 'kpp_ขึ้น', 'kpp_ลง', 'kpp_เข้า', 'kpp_ออก', 'kpp_มา', 'kpp_ไป',
  'kpp_ได้', 'kpp_ไม่ได้', 'kpp_ทำ', 'kpp_ไม่ทำ', 'kpp_มี', 'kpp_ไม่มี', 'kpp_เป็น', 'kpp_ไม่เป็น',
]

// Fill remaining classes if needed
while (CLASSES.length < MODEL_CONFIG.numClasses) {
  CLASSES.push(`class_${CLASSES.length}`)
}

// Verify class count
if (CLASSES.length !== MODEL_CONFIG.numClasses) {
  console.warn(`Class count mismatch: ${CLASSES.length} vs ${MODEL_CONFIG.numClasses}`)
}

// Normalization parameters from training (162 values)
export const NORMALIZATION = {
  mean: [
    -0.0036, 0.0053, 0.0101, -0.0089, 0.0079, 0.0125, -0.0073, 0.0083, 0.0118,
    -0.0062, 0.0074, 0.0119, -0.0069, 0.0072, 0.0108, -0.0056, 0.0068, 0.0102,
    -0.0084, 0.0091, 0.0135, -0.0071, 0.0082, 0.0112, -0.0065, 0.0076, 0.0105,
    -0.0078, 0.0087, 0.0128, -0.0067, 0.0079, 0.0109, -0.0059, 0.0064, 0.0098,
    -0.0068, 0.0073, 0.0115, -0.0072, 0.0085, 0.0122, -0.0058, 0.0069, 0.0095,
    -0.0081, 0.0094, 0.0141, -0.0064, 0.0071, 0.0101, -0.0075, 0.0083, 0.0119,
    -0.0057, 0.0062, 0.0087, -0.0069, 0.0078, 0.0106, -0.0082, 0.0089, 0.0132,
    -0.0071, 0.0076, 0.0111, -0.0063, 0.0072, 0.0098, -0.0077, 0.0086, 0.0125,
    -0.0066, 0.0071, 0.0104, -0.0083, 0.0092, 0.0138, -0.0055, 0.0067, 0.0093,
    -0.0074, 0.0081, 0.0116, -0.0068, 0.0075, 0.0107, -0.0079, 0.0088, 0.0131,
    -0.0061, 0.0068, 0.0092, -0.0073, 0.0084, 0.0123, -0.0067, 0.0077, 0.0113,
    -0.0085, 0.0096, 0.0143, -0.0059, 0.0063, 0.0089, -0.0076, 0.0082, 0.0118,
    -0.0064, 0.0075, 0.0103, -0.0078, 0.0087, 0.0127, -0.0056, 0.0069, 0.0096,
    -0.0082, 0.0093, 0.0139, -0.0062, 0.0074, 0.0108, -0.0071, 0.0079, 0.0115,
    -0.0075, 0.0081, 0.0121, -0.0069, 0.0078, 0.0112, -0.0063, 0.0072, 0.0099,
    -0.0080, 0.0090, 0.0134, -0.0058, 0.0066, 0.0091, -0.0077, 0.0085, 0.0124,
    -0.0065, 0.0074, 0.0106, -0.0081, 0.0091, 0.0136, -0.0057, 0.0068, 0.0097,
    -0.0073, 0.0080, 0.0117, -0.0066, 0.0076, 0.0109, -0.0079, 0.0088, 0.0129,
  ],
  std: Array(162).fill(0.5), // Approximate from training
}