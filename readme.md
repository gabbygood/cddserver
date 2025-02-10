gunicorn -w 4 -k uvicorn.workers.UvicornWorker plantdiseaseserver:app
uvicorn plantdiseaseserver:app --host localhost --port 8000 --reload

#let use the tflite lite model instead