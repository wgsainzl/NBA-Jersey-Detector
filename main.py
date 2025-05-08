import cv2
import numpy as np
import easyocr

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# NBA teams dictionary with LAB color values (more perceptual)
nba_teams = {
    "CLEVELAND": [120, 140, 138],
    "DALLAS": [125, 138, 111]
}

# --- Preprocessing for OCR ---
def preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    contrasted = cv2.convertScaleAbs(gray, alpha=2.5, beta=0)
    thresh = cv2.adaptiveThreshold(contrasted, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

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
    text_match = None
    for text in detected_texts:
        for team in nba_teams:
            if team in text or text in team:
                text_match = team
                break

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