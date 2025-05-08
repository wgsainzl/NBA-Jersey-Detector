import cv2
import numpy as np
import easyocr
import Levenshtein

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Use Levenshtein to catch simmilarities if text detector fail
def best_text_match(detected_texts, team_names):
    best_team = None
    best_score = float('inf')
    for text in detected_texts:
        for team in team_names:
            dist = Levenshtein.distance(text, team)
            if dist < best_score and dist <= 3: 
                best_score = dist
                best_team = team
    return best_team

# NBA teams dictionary with LAB color values (more perceptual)
nba_teams = {
    "CLEVELAND": [120, 140, 138],
    "DALLAS": [125, 138, 111]
}

# --- Preprocessing for OCR ---
def preprocess_for_ocr(roi):
    # Use the red channel — gold letters stand out more
    red_channel = roi[:, :, 2]

    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(red_channel)

    # Invert the image BEFORE thresholding: dark background becomes light
    inverted = cv2.bitwise_not(enhanced)

    # Adaptive threshold (text now is darker than background after inversion)
    thresh = cv2.adaptiveThreshold(
        inverted, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 10
    )

    return thresh

# --- Get Dominant Color in BGR and convert to LAB ---
def get_dominant_color(image):
    pixels = image.reshape(-1, 3)
    avg_bgr = np.mean(pixels, axis=0)
    avg_bgr = np.uint8([[avg_bgr]])
    avg_lab = cv2.cvtColor(avg_bgr, cv2.COLOR_BGR2LAB)[0][0]
    return avg_bgr[0][0].tolist(), avg_lab.tolist()

# --- Match LAB color to team ---
def match_color(lab_color, team_colors):
    min_dist = float('inf')
    matched_team = None
    for team, ref_color in team_colors.items():
        dist = np.linalg.norm(np.array(lab_color) - np.array(ref_color))
        if dist < min_dist:
            min_dist = dist
            matched_team = team
    return matched_team

# --- Start Webcam ---
# For Mac:
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
# For Windows, use:
# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    roi = frame[h//3:h//2, w//4:w*3//4]
    cv2.rectangle(frame, (w//4, h//3), (w*3//4, h//2), (0, 255, 0), 2)

    # --- OCR ---
    ocr_input = preprocess_for_ocr(roi)
    results = reader.readtext(ocr_input)
    detected_texts = [res[1].upper() for res in results]
    joined_text = ", ".join(detected_texts) if detected_texts else "None"

    # --- Dominant color ---
    dominant_bgr, dominant_lab = get_dominant_color(roi)
    color_match = match_color(dominant_lab, nba_teams)

    # --- Text match ---
    text_match = best_text_match(detected_texts, nba_teams.keys())

    # --- Final decision ---
    final_team = None
    if text_match == color_match:
        final_team = text_match
    elif text_match:
        final_team = text_match + " (?)"
    elif color_match:
        final_team = color_match + " (color only)"

    # --- Display ---
    cv2.putText(frame, f"Detected: {final_team if final_team else 'None'}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"OCR Text: {joined_text}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"BGR Color: {tuple(int(c) for c in dominant_bgr)}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)
    cv2.putText(frame, f"LAB Color: {tuple(int(c) for c in dominant_lab)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)

    cv2.imshow("NBA Jersey Detector", frame)
    cv2.imshow("OCR Input", ocr_input)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()