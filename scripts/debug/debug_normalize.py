# Debug the normalize_label function
def normalize_label(label):
    """Normalize label by removing suffixes like _สองมือ_6, _อายุเท่ากันหรือน้อยกว่า_5, etc."""
    if not label:
        return None
    
    # Filter out null samples
    if label.startswith('null_'):
        return None
    
    # Split on common suffixes and get base word
    for suffix in ['_สองมือ_', '_อายุเท่ากันหรือน้อยกว่า_', '_ทำท่ามือถามไปยังผู้นั้น_', 
                   '_เปิดมือสองข้าง_', '_บุลคคลที่สาม_', '_var_']:
        if suffix in label:
            return label.split(suffix)[0]
    
    return label

# Test with sample filenames
test_labels = [
    "น้ำ_var_1_10",
    "กรุงเทพ_var_1_0", 
    "ร้อน_สองมือ_6",
    "ขอโทษ_อายุเท่ากันหรือน้อยกว่า_9",
    "ถาม_ทำท่ามือถามไปยังผู้นั้น_5",
    "null_fidgeter_2",
    "ขอบคุณ_เปิดมือสองข้าง_6",
    "เธอ_บุลคคลที่สาม_0"
]

print("Testing normalize_label:")
for label in test_labels:
    result = normalize_label(label)
    print(f"  {label} -> {result}")