pip install -r requirements.txt

export YOLO_MODEL_PATH="/kaggle/working/Auto-WCEBleedGen/
python server.py --host 0.0.0.0 --port 5000 --fps 5 --conf 0.25
