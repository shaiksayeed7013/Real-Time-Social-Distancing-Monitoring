import numpy as np
import time
import cv2
import math
import imutils
import pyttsx3

# Load labels file
labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# Random colors
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Weights path
weightsPath = "D:\AI&ML\social-distance-detector-master\yolov3.weights"
configPath = "D:\AI&ML\social-distance-detector-master\yolov3.cfg"

# Text-to-speech engine
engine = pyttsx3.init()

# Print loading message
print("Loading Machine Learning Model ...")

# Read model
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Start camera
print("Starting Camera ...")
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, image = cap.read()
    image = imutils.resize(image, width=800)
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()

    # Get Unconnected Out Layers
    unconnected_layers = net.getUnconnectedOutLayers()
    ln = [ln[int(layer) - 1] for layer in unconnected_layers]

    # Resize input image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Prediction time/frame : {:.6f} seconds".format(end - start))

    # Detect objects
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    color_far = (0, 255, 0)
    color_close = (0, 0, 255)

    if len(idxs) > 0:
        for i in idxs.flatten():
            for j in idxs.flatten():
                if i < j:
                    x_dist = boxes[j][0] - boxes[i][0]
                    y_dist = boxes[j][1] - boxes[i][1]
                    distance_between_objects = math.sqrt(x_dist * x_dist + y_dist * y_dist)

                    if distance_between_objects < 220:
                        cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), color_close, 2)
                        cv2.rectangle(image, (boxes[j][0], boxes[j][1]), (boxes[j][0] + boxes[j][2], boxes[j][1] + boxes[j][3]), color_close, 2)
                        cv2.putText(image, "Red Alert: MOVE AWAY", (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_close, 2)
                        cv2.putText(image, "Red Alert: MOVE AWAY", (boxes[j][0], boxes[j][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_close, 2)
                        
                        # Text-to-speech
                        engine.say("Please maintain social distancing")
                        engine.runAndWait()
                    else:
                        cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), color_far, 2)
                        cv2.rectangle(image, (boxes[j][0], boxes[j][1]), (boxes[j][0] + boxes[j][2], boxes[j][1] + boxes[j][3]), color_far, 2)
                        cv2.putText(image, "Normal", (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_far, 2)
                        cv2.putText(image, "Normal", (boxes[j][0], boxes[j][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_far, 2)

    cv2.imshow("Social Distancing Detector", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
