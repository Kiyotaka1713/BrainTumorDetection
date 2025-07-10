

from ultralytics import YOLO
import yaml
if __name__ == '__main__':
    model = YOLO("FYPv10CM_model.pt")
    metrics = model.val(data="dataset.yaml",
                        imgsz=640,
                         batch=16,  # Batch size (adjust based on GPU capacity)
                         workers=4,  # Number of data loader workers
                    )

# Print metrics
    print(metrics)