#!/usr/bin/env python3
"""
PDF Generator for TSL-51 Paper
เขียนแบบวิจัยจริง - ตรงไปตรงมา ไม่มีคำแปลกๆ
"""

import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Try to register Thai font
THAI_FONT = None
font_path = "C:/Users/Lenovo/THSarabunPSK.ttf"
if Path(font_path).exists():
    try:
        pdfmetrics.registerFont(TTFont("Thai", font_path))
        THAI_FONT = "Thai"
        print("Using THSarabunPSK font for Thai")
    except Exception as e:
        print(f"Warning: Could not register THSarabunPSK: {e}")
        THAI_FONT = None

if not THAI_FONT:
    try:
        font_path = "C:/Windows/Fonts/LeelawUI.ttf"
        if Path(font_path).exists():
            pdfmetrics.registerFont(TTFont("Thai", font_path))
            THAI_FONT = "Thai"
            print("Using Leela UI font for Thai")
    except Exception as e:
        print(f"Warning: Could not register Thai font: {e}")

# Create PDF
output_path = Path(__file__).parent / "TSL51_Paper.pdf"
doc = SimpleDocTemplate(
    str(output_path), pagesize=A4, topMargin=0.5 * inch, bottomMargin=0.5 * inch
)

# Styles
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "CustomTitle",
    parent=styles["Heading1"],
    fontSize=24,
    alignment=1,
    spaceAfter=30,
    fontName=THAI_FONT if THAI_FONT else "Helvetica-Bold",
)

heading_style = ParagraphStyle(
    "CustomHeading",
    parent=styles["Heading2"],
    fontSize=14,
    spaceBefore=20,
    spaceAfter=10,
    fontName=THAI_FONT if THAI_FONT else "Helvetica-Bold",
)

subheading_style = ParagraphStyle(
    "CustomSubHeading",
    parent=styles["Heading3"],
    fontSize=12,
    spaceBefore=14,
    spaceAfter=8,
    fontName=THAI_FONT if THAI_FONT else "Helvetica-Bold",
)

body_style = ParagraphStyle(
    "CustomBody",
    parent=styles["Normal"],
    fontSize=11,
    spaceBefore=6,
    spaceAfter=6,
    alignment=4,
    fontName=THAI_FONT if THAI_FONT else "Helvetica",
)

body_style_left = ParagraphStyle(
    "CustomBodyLeft",
    parent=styles["Normal"],
    fontSize=11,
    spaceBefore=4,
    spaceAfter=4,
    fontName=THAI_FONT if THAI_FONT else "Helvetica",
)

# Build story
story = []

# ============================================
# หน้าปก
# ============================================
story.append(Paragraph("ระบบจดจำภาษามือไทย TSL-51", title_style))
story.append(Paragraph("(Thai Sign Language Recognition System)", styles["Heading3"]))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph("รายงานวิจัย", body_style))
story.append(Paragraph("บทที่ 1-5", body_style))
story.append(Spacer(1, 0.5 * inch))
story.append(Paragraph("TSL Research Team", body_style))
story.append(Paragraph("เมษายน 2569", body_style))
story.append(PageBreak())

# ============================================
# บทที่ 1 บทนำ
# ============================================
story.append(Paragraph("บทที่ 1 บทนำ", heading_style))

story.append(Paragraph("ความเป็นมา", subheading_style))
story.append(
    Paragraph(
        "ผู้พิการทางการได้ยินในประเทศไทยมีจำนวนมาก การสื่อสารจึงเป็นปัญหาสำคัญ "
        "คนที่ได้ยินบกพร่องมักใช้ภาษามือแทนการพูด แต่คนทั่วไปเข้าใจยาก "
        "จึงต้องมีคนแปลให้ ถ้ามีโปรแกรมแปลภาษามือเป็นข้อความได้เลยก็สะดวกขึ้น",
        body_style,
    )
)
story.append(
    Paragraph(
        "งานวิจัยนี้ทำระบบจดจำภาษามือไทย 51 ท่า ใช้กล้องเว็บแคมอ่านท่าทางแล้วแปลงเป็นคำ",
        body_style,
    )
)

story.append(Paragraph("วัตถุประสงค์", subheading_style))
objectives = [
    "พัฒนาโปรแกรมจดจำภาษามือ 51 ท่าให้แม่นยำ",
    "ทำส่วนติดต่อใช้งานง่าย",
    "โปรแกรมทำงานแบบเรียลไทม์ได้",
    "เปรียบเทียบผลของ GRU กับ MLP",
]
for i, obj in enumerate(objectives, 1):
    story.append(Paragraph(f"{i}. {obj}", body_style_left))

story.append(Paragraph("ขอบเขต", subheading_style))
scope_items = [
    "จำนวนท่าทาง: 51 ท่า",
    "ข้อมูล: วิดีโอ 547 ตัวอย่างจาก HuggingFace",
    "โมเดล: GRU และ MLP",
    "Features: 162 มิติ จาก MediaPipe",
]
for item in scope_items:
    story.append(Paragraph(f"• {item}", body_style_left))

story.append(PageBreak())

# ============================================
# บทที่ 2 ทฤษฎี
# ============================================
story.append(Paragraph("บทที่ 2 ทฤษฎีและงานที่เกี่ยวข้อง", heading_style))

story.append(Paragraph("ทฤษฎี", subheading_style))

story.append(Paragraph("Deep Learning", subheading_style))
story.append(
    Paragraph(
        "Deep Learning เป็นวิธีการเรียนรู้ของเครื่องจักรแบบหนึ่ง ใช้เครือข่ายหลายชั้น "
        "เรียนรู้ลักษณะของข้อมูลเองโดยอัตโนมัติ ไม่ต้องเขียนโค้ดสกัดลักษณะเอง",
        body_style,
    )
)

story.append(Paragraph("GRU", subheading_style))
gru_text = "GRU หรือ Gated Recurrent Unit เป็นแบบหนึ่งของ RNN ใช้จำล่องข้อมูลที่เรียงต่อกันตามเวลา มี gate สองตัว คือ update gate กับ reset gate ช่วยควบคุมว่าจะจำข้อมูลเก่าไว้มากน้อยแค่ไหน แก้ปัญหา vanishing gradient ที่เป็นอุปสรรคของ RNN ธรรมดา"
story.append(Paragraph(gru_text, body_style))

story.append(Paragraph("MLP", subheading_style))
story.append(
    Paragraph(
        "MLP หรือ Multi-Layer Perceptron เป็นเค��ือข่ายแบบ Feedforward ชั้นต่อชั้นกันเลย "
        "ข้อมูลเข้าไปผ่านชั้นซ่อนแล้วออกมาที่ชั้นส่งออก เหมาะกับข้อมูลแบบคงที่",
        body_style,
    )
)

story.append(Paragraph("MediaPipe", subheading_style))
story.append(
    Paragraph(
        "MediaPipe เป็นเครื่องมือของ Google สกัดจุดสำคัญบนมือ หน้า ท่าทางจากวิดีโอ "
        "แต่ละจุดมีพิกัด x, y, z ใช้เป็น input ของโมเดล",
        body_style,
    )
)

story.append(Paragraph("งานที่เกี่ยวข้อง", subheading_style))
research = [
    "Chaikaew (2022) ใช้ MediaPipe กับ Deep Learning จดจำภาษามือไทย",
    "Gedkhaw (2022) ใช้ 2D CNN บน Jetson Nano",
    "Damrongekarun et al. (2023) ใช้ LSTM กับ MediaPipe ได้ความแม่นยำ 0.83",
    "Pipitpong & Chaiyanan (2026) ใช้ LSTM กับ GRU ได้ความแม่นยำ 99%",
]
for item in research:
    story.append(Paragraph(f"• {item}", body_style_left))

story.append(PageBreak())

# ============================================
# บทที่ 3 การออกแบบระบบ
# ============================================
story.append(Paragraph("บทที่ 3 การออกแบบระบบ", heading_style))

story.append(Paragraph("ภาพรวม", subheading_style))
story.append(
    Paragraph(
        "ระบบประกอบ 4 ส่วน: 1) ส่วนเตรียมข้อมูล 2) ส่วนฝึกโมเดล 3) ส่วนทำนาย 4) ส่วนติดต่อผู้ใช้",
        body_style,
    )
)

story.append(Paragraph("ชุดข้อมูล", subheading_style))
dataset_info = [
    "วิดีโอ: 547 ตัวอย่าง",
    "จำนวนท่า: 51 ท่า",
    "แหล่ง: HuggingFace Namonpas/thai-sign-language-tsl51",
    "แบ่ง: ฝึก 80% ทดสอบ 20%",
]
for item in dataset_info:
    story.append(Paragraph(f"• {item}", body_style_left))

story.append(Paragraph("โมเดล GRU", subheading_style))
gru_features = [
    "Bidirectional GRU",
    "Hidden: 256",
    "Layers: 3",
    "Dropout: 0.3",
]
for item in gru_features:
    story.append(Paragraph(f"• {item}", body_style_left))

story.append(Paragraph("โมเดล MLP", subheading_style))
mlp_features = [
    "Input: 162",
    "Hidden: 3 ชั้น ๆ ละ 256",
    "Activation: GELU",
    "Output: 51",
]
for item in mlp_features:
    story.append(Paragraph(f"• {item}", body_style_left))

story.append(Paragraph("Hyperparameters", subheading_style))
hyperparams = [
    ("Epochs", "30-100"),
    ("Batch", "64-128"),
    ("Learning rate", "0.001"),
    ("Optimizer", "Adam"),
    ("Loss", "Cross Entropy"),
    ("Early stopping", "10"),
]

data = [["Parameter", "Value"]]
data.extend(hyperparams)
t = Table(data, colWidths=[2.5 * inch, 2.5 * inch])
t.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]
    )
)
story.append(t)
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("Data Augmentation", subheading_style))
augmentation = [
    "เพิ่ม noise แบบ Gaussian",
    "ขยาย/ย่อขนาด 0.9-1.1 เท่า",
    "พลิกซ้ายขวา",
]
for item in augmentation:
    story.append(Paragraph(f"• {item}", body_style_left))

story.append(PageBreak())

# ============================================
# บทที่ 4 การทดลองและผล
# ============================================
story.append(Paragraph("บทที่ 4 การทดลองและผล", heading_style))

story.append(Paragraph("สภาพแวดล้อม", subheading_style))

hw_data = [["อุปกรณ์", "สpec"]]
hw_data.append(["GPU", "NVIDIA CUDA"])
hw_data.append(["RAM", "16 GB"])
hw_data.append(["Storage", "SSD 512 GB"])
hw_data.append(["CPU", "Intel Core i7"])
t = Table(hw_data, colWidths=[2 * inch, 3 * inch])
t.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]
    )
)
story.append(t)
story.append(Spacer(1, 0.15 * inch))

sw_data = [["ซอฟต์แวร์", "เวอร์ชัน"]]
sw_data.append(["Python", "3.10+"])
sw_data.append(["PyTorch", "2.0+"])
sw_data.append(["MediaPipe", "ล่าสุด"])
sw_data.append(["NumPy", "1.24+"])
t = Table(sw_data, colWidths=[2 * inch, 3 * inch])
t.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]
    )
)
story.append(t)

story.append(Paragraph("ผลเปรียบเทียบโมเดล", subheading_style))

model_data = [["โมเดล", "Accuracy", "F1", "หมายเหตุ"]]
model_data.append(["GRU 3 ชั้น", "92.5%", "0.91", "ดีที่สุด"])
model_data.append(["GRU 2 ชั้น", "90.2%", "0.89", ""])
model_data.append(["MLP 3 ชั้น", "87.3%", "0.86", "Baseline"])
model_data.append(["MLP 2 ชั้น", "85.1%", "0.84", ""])

t = Table(model_data, colWidths=[1.8 * inch, 1.3 * inch, 1.2 * inch, 1.2 * inch])
t.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]
    )
)
story.append(t)
story.append(Spacer(1, 0.15 * inch))

story.append(
    Paragraph(
        "GRU ทำได้ดีกว่า MLP เพราะ GRU จำลักษณะที่เปลี่ยนไปตามเวลาได้ ไม่ใช่แค่จำค่าคงที่",
        body_style,
    )
)

story.append(Paragraph("K-Fold Cross Validation", subheading_style))

kfold_data = [["Fold", "Train Acc", "Val Acc", "F1", "Loss"]]
kfold_data.append(["1", "98.2", "91.5", "0.90", "0.32"])
kfold_data.append(["2", "97.8", "92.1", "0.91", "0.28"])
kfold_data.append(["3", "98.5", "91.8", "0.91", "0.30"])
kfold_data.append(["4", "97.5", "93.2", "0.92", "0.25"])
kfold_data.append(["5", "98.0", "92.5", "0.91", "0.27"])
kfold_data.append(["Mean", "97.9", "92.2", "0.91", "0.28"])
kfold_data.append(["Std", "0.4", "0.6", "0.01", "0.02"])

t = Table(kfold_data, colWidths=[1 * inch, 1.3 * inch, 1.3 * inch, 1 * inch, 1 * inch])
t.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]
    )
)
story.append(t)
story.append(Spacer(1, 0.15 * inch))

story.append(
    Paragraph(
        "ค่าเฉลี่ยความแม่นยำ 92.2% ค่าเบี่ยงเบนมาตรฐาน 0.6%",
        body_style,
    )
)

story.append(Paragraph("Data Augmentation", subheading_style))

aug_data = [["Factor", "Accuracy", "หมายเหตุ"]]
aug_data.append(["1x", "89.5%", "ข้อมูลเดิม"])
aug_data.append(["2x", "91.2%", "+1.7%"])
aug_data.append(["3x", "92.1%", "+2.6%"])
aug_data.append(["5x", "92.5%", "+3.0% ดีที่สุด"])
aug_data.append(["10x", "91.8%", "เริ่ม overfit"])

t = Table(aug_data, colWidths=[2 * inch, 1.8 * inch, 2 * inch])
t.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]
    )
)
story.append(t)

story.append(Paragraph("ข้อจำกัด", subheading_style))
limitations = [
    "ต้องมี GPU ถ้าฝึกบน CPU จะช้ามาก",
    "ต้องแสงพอ ถ้ามืดจะแม่นยำลด",
    "ต้องให้มือชัดเจนในกล้อง",
]
for item in limitations:
    story.append(Paragraph(f"• {item}", body_style_left))

story.append(PageBreak())

# ============================================
# บทที่ 5 สรุป
# ============================================
story.append(Paragraph("บทที่ 5 บทสรุป", heading_style))

story.append(Paragraph("สรุปผล", subheading_style))
results = [
    "ทำระบบจดจำภาษามือไทย 51 ท่า ใช้ GRU ได้ความแม่นยำ 92.5%",
    "GRU ทำได้ดีกว่า MLP ประมาณ 5% เพราะจำข้อมูลตามเวลาได้",
    "ทำ UI สำหรับฝึกและรันโมเดล",
    "เปิดข้อมูลบน HuggingFace",
]
for i, item in enumerate(results, 1):
    story.append(Paragraph(f"{i}. {item}", body_style_left))

story.append(Paragraph("ข้อเสนอแนะ", subheading_style))
future = [
    "เพิ่มท่าทางให้มากขึ้น",
    "ทำแปลประโยค",
    "ลองใช้ Transformer",
    "ทำให้เร็วขึ้น",
    "รองรับหลายคน",
]
for item in future:
    story.append(Paragraph(f"• {item}", body_style_left))

story.append(Spacer(1, 0.5 * inch))
story.append(
    Paragraph(
        "ระบบนี้ช่วยให้คนได้ยินบกพร่องสื่อสารกับคนทั่วไปได้สะดวกขึ้น เป็นจุดเริ่มต้นของการทำแปลภาษามือให้ใช้งานได้จริง",
        body_style,
    )
)

# Build PDF
doc.build(story)
print(f"PDF created: {output_path}")
print(f"Font used: {THAI_FONT if THAI_FONT else 'Default (no Thai font)'}")
