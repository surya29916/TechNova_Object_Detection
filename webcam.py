

from ultralytics import YOLO
import cv2

# Step 1: Load the pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Step 2: Open the webcam
cap = cv2.VideoCapture(0)

# Step 3: Check whether the webcam opened
if not cap.isOpened():
    print("Error: Could not open webcam!")
    raise SystemExit

print("Webcam started!")
print("Press 'q' to stop detection.")

# Step 4: Detect objects continuously
while True:
    success, frame = cap.read()

    if not success:
        print("Error: Could not read webcam frame.")


    # Step 5: Detect objects in the current frame
    results = model(frame, verbose=False)

    # Step 6: Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Step 7: Count detected objects in this frame
    object_counts = {}

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]

        object_counts[name] = object_counts.get(name, 0) + 1

    # Step 8: Display object counts on the frame
    y_position = 30

    for name, count in object_counts.items():
        text = f"{name}: {count}"
        cv2.putText(
            annotated_frame,
            text,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y_position += 30

    # Step 9: Show the webcam video
    cv2.imshow("TechNova Real-Time Object Detection", annotated_frame)

    # Step 10: Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Step 11: Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

print("Webcam detection stopped.")