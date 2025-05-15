import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.4

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    persons = [x for x in results.pred[0] if int(x[5]) == 0]

    for *xyxy, conf, cls in persons:
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(
            xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"Pessoa {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detecção de Pessoas - Protótipo", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
