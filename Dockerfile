# Gunakan base image python yang ringan
FROM python:3.10-slim

# Set environment variables
ENV PYTHONBUFFERED True
ENV APP_HOME /app
ENV PORT 8080

# Buat dan set working directory
WORKDIR $APP_HOME

# Salin file requirements.txt dan install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Salin seluruh aplikasi ke dalam container
COPY . ./

# Jalankan aplikasi menggunakan gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
