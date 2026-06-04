# TSL-51 Webcam Playbook

เปิดกล้องแล้วทำนายถูกคำ — checklist สั้นๆ กันทำผิด

## ก่อนเริ่ม (ครั้งเดียว)

```powershell
cd D:\TSL
D:\TSL\.venv-train\Scripts\python.exe scripts\tsl51_doctor.py --run-smoke
```

- ต้องได้ `baseline_live_top1: 1.0` ใน smoke — ถ้าไม่ใช่ **restart API** ด้วย venv ที่ถูก
- อ่าน `[NEXT]` ที่ doctor พิมพ์ — ทำขั้นนั้นอย่างเดียว

## Phase A — เปิดระบบ

```powershell
powershell -File scripts\tsl_translate_dev.ps1
```

เปิด http://localhost:3000 แล้วตั้งค่า:

| การตั้งค่า | ค่าที่ถูก |
|---|---|
| Track | TSL-51 |
| Artifact | `full51_v3_external_weighted` (51 คลาส) |
| Preset | **สมดุล** (แนะนำ · หลายคำในประโยค) · หรือ **แม่นยำ** ถ้าท่าเดี่ยวยังผิด |

**วิธีเซ็น:** เซ็นท่าให้มือเห็นชัด → **หยุดนิ่งสั้นๆ** หลังจบท่า (~0.3 วิ สมดุล · ~0.5 วิ แม่นยำ)

## Phase B — อัด holdout จากกล้องจริง (ถ้ายังทำนายผิด)

```powershell
D:\TSL\.venv-train\Scripts\python.exe scripts\record_webcam_holdout.py `
  --out-dir data\webcam_holdout `
  --labels .tools\tsl51_experiments\full51_v3_external_weighted\artifacts\tsl51\tsl51_labels.json `
  --signs "0:15" --num-clips 3 --record-s 5
```

แสงสม่ำเสมอ · กล้องระดับหน้า · เซ็นช้า ~5 วิ/คลิป

## Phase C — วัดผล

```powershell
D:\TSL\.venv-train\Scripts\python.exe scripts\evaluate_webcam_holdout.py
D:\TSL\.venv-train\Scripts\python.exe scripts\run_webcam_holdout_diagnosis.py
D:\TSL\.venv-train\Scripts\python.exe scripts\check_webcam_holdout_gate.py
```

| Top-1 holdout | ทำอะไร |
|---|---|
| ≥ 70% | ใช้งาน live ได้ — จูน threshold เล็กน้อยถ้าจำเป็น |
| 50–70% | อัดคลิปเพิ่ม + fine-tune |
| < 50% | Fine-tune (Phase D) |

## Phase D — Fine-tune (เมื่อ holdout < 70%)

1. อัด train clips (แยกจาก holdout):

```powershell
D:\TSL\.venv-train\Scripts\python.exe scripts\record_webcam_holdout.py `
  --out-dir data\webcam_train --split train --signs "0:15"
```

2. รัน pipeline:

```powershell
D:\TSL\.venv-train\Scripts\python.exe scripts\run_webcam_finetune.py `
  --base-cache <path\to\tsl51_features.npz>
```

3. โหลด artifact ใหม่ใน UI → restart API

## Phase E — ยืนยัน live

```powershell
D:\TSL\.venv-train\Scripts\python.exe scripts\diagnose_live_vs_clip.py `
  --samples data\webcam_holdout\webcam_holdout.csv `
  --out-dir reports\live_vs_clip_diagnosis\holdout_live
```

Live committed Top-1 ควรใกล้ offline ≥ 80%

## ห้ามทำ

- ใช้ preset **เร็ว** แล้วคาดหวังความแม่นยำ
- ใช้ legacy `artifacts/tsl51` (47 คลาส)
- Flip ภาพก่อนส่ง inference
- Retrain ก่อนมี holdout จากกล้องจริง
- ลืม restart API หลังแก้ `inference.py`

## รายงานอ้างอิง

- `reports/live_vs_clip_diagnosis/root_cause_summary.json`
- `reports/tsl51_doctor.json`
- `reports/webcam_holdout_eval/tuning_suggestions.json`
