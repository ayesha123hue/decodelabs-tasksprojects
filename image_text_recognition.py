# Project 4: Image or Text Recognition (Basic)
# DecodeLabs - Industrial Training Kit
# Goal: Image ke andar likhi hui text ko machine se pehchanwana (OCR)

import pytesseract
from PIL import Image
import cv2

# ---------------------------------------------------------
# STEP 1: INPUT - Image load karo
# ---------------------------------------------------------
image_path = "sample_text.png"   # Apni image ka naam/path yahan daalo
image = Image.open(image_path)

print("Image loaded:", image_path)
print("Image size:", image.size)
print("-" * 50)

# ---------------------------------------------------------
# STEP 2: PROCESS - OpenCV se image ko clean karo (preprocessing)
# ---------------------------------------------------------
# OpenCV format mein load karo
cv_image = cv2.imread(image_path)

# Grayscale mein convert karo - OCR grayscale par behtar kaam karta hai
gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# Thresholding - black & white saaf karne ke liye
_, processed_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# Processed image save karo (dekhne ke liye)
cv2.imwrite("processed_image.png", processed_image)

# ---------------------------------------------------------
# STEP 3: OUTPUT - pytesseract se text nikaalo (OCR)
# ---------------------------------------------------------
extracted_text = pytesseract.image_to_string(Image.open("processed_image.png"))

print("Extracted Text:")
print(extracted_text.strip())
print("-" * 50)

# ---------------------------------------------------------
# BONUS: Confidence score bhi dekho (kitna sure hai model)
# ---------------------------------------------------------
data = pytesseract.image_to_data(Image.open("processed_image.png"), output_type=pytesseract.Output.DICT)

print("Word-by-word confidence:")
for i, word in enumerate(data["text"]):
    if word.strip():   # khaali words skip karo
        confidence = data["conf"][i]
        print(f"  '{word}' -> Confidence: {confidence}%")
