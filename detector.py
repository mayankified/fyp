# detector.py
import config

class YOLODetector:
    def __init__(self):
        self.model = None
        if config.USE_AI:
            try:
                from ultralytics import YOLO
                print(f"Loading model: {config.MODEL_PATH}")
                self.model = YOLO(config.MODEL_PATH)
            except ImportError:
                print("WARNING: Ultralytics not found. AI disabled.")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("AI Detection is DISABLED in config.")

    def predict(self, img_bgr):
        # Pass-through if model isn't ready
        if self.model is None:
            return img_bgr

        try:
            import cv2
            results = self.model.predict(source=img_bgr, conf=config.CONF_THRESH, verbose=False)
            annotated = results[0].plot()
            return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        except Exception:
            return img_bgr