# Базовый образ
FROM python:3.11-slim

# Установка системных зависимостей для OpenCV и сборки пакетов
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    git \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Обновление pip и установка зависимостей
RUN pip install --upgrade pip setuptools wheel

# Копируем зависимости
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Установка PyTorch из официального индекса (CPU версия для меньшего размера)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Установка остальных зависимостей из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект внутрь контейнера
COPY . /app

# Устанавливаем рабочую директорию
WORKDIR /app

# Команда запуска FastAPI
# Используем переменную PORT из окружения, если доступна (для Railway, Render и др.)
# По умолчанию используем порт 8000
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
