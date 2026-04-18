# Check the raw bytes of a known Thai word
# "เช้า" in UTF-8 should be: e0 b9 8a e0 b9 94 e0 b8 b2
# Let's check what the actual bytes are

from huggingface_hub import hf_hub_download

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/expert_metadata.csv',
    repo_type='dataset'
)

# Read CSV as binary and find where "เช้า" should be
with open(meta_path, 'rb') as f:
    content = f.read()

# Look for "เช้า" pattern in bytes
thai_chao = "เช้า".encode('utf-8')
print(f"'เช้า' in UTF-8 hex: {thai_chao.hex()}")

# Search for this pattern in the raw content
if thai_chao in content:
    print("Found UTF-8 'เช้า' in file!")
else:
    print("UTF-8 'เช้า' NOT found")
    
# Try different encodings
    import logging
    logger = logging.getLogger(__name__)

    for enc_name, enc_obj in [('cp874', 'cp874'), ('tis620', 'tis620')]:
        try:
            encoded = "เช้า".encode(enc_name)
            print(f"'{enc_name}' hex: {encoded.hex()}")
            if encoded in content:
                print("  Found in file!")
        except Exception as e:
            logger.debug("Encoding %s not available on this system: %s", enc_name, e)

# Let's look at what the actual bytes are around sign_clean field
# Find position of a known video_id
search = b'vid_1313_blur_k3,'
pos = content.find(search)
if pos > 0:
    # Get surrounding bytes
    start = pos
    end = min(pos + 100, len(content))
    snippet = content[start:end]
    print("\nRaw bytes around vid_1313_blur_k3:")
    print(f"  {snippet[:50]}")
    print(f"  Hex: {snippet[:50].hex()}")
