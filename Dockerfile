# Используйте базовый образ Python
FROM python:3.11

# Установите зависимости
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Скопируйте текущий каталог в контейнер в каталог app
COPY . /app

# Установите рабочую директорию в /app
WORKDIR /app

# Запустите ваш скрипт при запуске контейнера
CMD ["python", "test.py"]
