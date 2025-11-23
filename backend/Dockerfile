FROM python:3.11-slim

# Install OpenCV + image libs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy backend/app into container
COPY app ./app

# make sure model filename matches
ENV ENCRYPTIC_MODEL_PATH=/app/app/model/model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
