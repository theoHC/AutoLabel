from ultralytics import YOLO

model = YOLO("yolo26l-obb.pt")  # load pretrained model
model.train(data="perfpro_dataset/dataset.yaml",
            name="perfpro", epochs=30, imgsz=640)
