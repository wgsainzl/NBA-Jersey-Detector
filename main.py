import cv2
import numpy as np
import easyocr

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# NBA teams dictionary with dominant BGR color
nba_teams = {
    "CLEVELAND": (111, 38, 61),
    "DALLAS": (0, 83, 188)
}

# Helper: Calculate dominant color in region
def get_dominant_color(image):
    pixels = image.reshape(-1, 3)
    avg_color = np.mean(pixels, axis=0)
    return tuple(int(c) for c in avg_color)

# Helper: Find best color match
def match_color(color, team_colors):
    min_dist = float('inf')
    matched_team = None
    for team, ref_color in team_colors.items():
        dist = np.linalg.norm(np.array(color) - np.array(ref_color))
        if dist < min_dist:
            min_dist = dist
            matched_team = team
    return matched_team

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame = cv2.resize(frame, (640, 480))

    # Define ROI for jersey detection (center)
    h, w, _ = frame.shape
    roi = frame[h//3:h//2, w//4:w*3//4]
    cv2.rectangle(frame, (w//4, h//3), (w*3//4, h//2), (0,255,0), 2)

    # OCR
    results = reader.readtext(roi)
    detected_texts = [res[1].upper() for res in results]

    # Dominant color
    dominant_color = get_dominant_color(roi)

    # Match color
    color_match = match_color(dominant_color, nba_teams)

    # Match text
    text_match = None
    for text in detected_texts:
        for team in nba_teams:
            if team in text or text in team:
                text_match = team
                break

    # Final decision
    final_team = None
    if text_match == color_match:
        final_team = text_match
    elif text_match:
        final_team = text_match + " (?)"
    elif color_match:
        final_team = color_match + " (color only)"

    # Display result
    if final_team:
        cv2.putText(frame, f"Detected: {final_team}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("NBA Jersey Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
