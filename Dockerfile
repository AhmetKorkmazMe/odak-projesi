# Temel imaj olarak Python 3.10 kullanıyoruz.
FROM python:3.10-slim-bookworm

# --- GÜNCELLENDİ ---
# Türkçe karakterler için font paketi eklendi.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    tesseract-ocr-tur \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini belirliyoruz.
WORKDIR /app

# Gerekli bağımlılıklar dosyasını kopyalıyoruz.
COPY requirements.txt .

# Bağımlılıkları yüklüyoruz.
RUN pip install --no-cache-dir -r requirements.txt

# Tüm uygulama dosyalarını kopyalıyoruz.
COPY . .

# Flask uygulamasının çalışacağı portu belirliyoruz.
EXPOSE 5000

# Flask uygulamasını başlatma komutu.
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
