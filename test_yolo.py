from ultralytics import YOLO

# load pretrained model.
model: YOLO = YOLO(model="yolov8n.pt")

# inference
# save flgをTrueにすることで推論結果を描画した画像を保存できる。
result: list = model.predict("./images/bus.jpg", save=True)