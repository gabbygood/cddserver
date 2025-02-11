gunicorn -w 4 -k uvicorn.workers.UvicornWorker plantdiseaseserver:app
uvicorn plantdiseaseserver:app --host 192.168.229.213 --port 8000 --reload

#let use the tflite lite model instead

### New Live Successfull
https://cddserver.onrender.com

### API Endpoint
https://cddserver.onrender.com/predict


### Remote From Laptop
