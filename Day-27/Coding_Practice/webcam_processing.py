
import cv2

# ==========================================
# 1. Initialize webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access webcam.")
    exit()

print("Webcam started successfully!")
print("Press Q to exit.")

# ==========================================
# 2. Capture and process frames
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Convert frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    # Display original webcam footage
    cv2.imshow("Live Webcam", frame)

    # Display processed footage
    cv2.imshow("Live Canny Edge Detection", edges)

    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# 3. Release webcam
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Webcam stopped successfully!")
