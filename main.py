
from ultralytics import YOLO
import cv2

# Step 1: Load the pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Step 2: Load the sample image
image = cv2.imread("bus.jpg")

# Step 3: Check whether the image loaded successfully
if image is None:
    print("Error: bus.jpg not found!")
    raise SystemExit

# Step 4: Detect objects in the image
results = model(image)

# Step 5: Draw bounding boxes and labels
annotated_image = results[0].plot()

# Step 6: Display the detected image
cv2.imshow("TechNova Object Detection", annotated_image)

# Step 7: Print detected objects and confidence scores
print("\nDetected Objects:")

object_counts = {}

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = model.names[class_id]

        print(f"{name}: {confidence * 100:.2f}%")

        # Count each detected object
        if name in object_counts:
            object_counts[name] += 1
        else:
            object_counts[name] = 1

# Step 8: Display the object counts
print("\nObject Count:")

for name, count in object_counts.items():
    print(f"{name}: {count}")

# Step 9: Display the total number of detected objects
total_objects = sum(object_counts.values())
print(f"\nTotal objects: {total_objects}")

# Step 10: Close the image window when a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()