# 1️⃣ Base image (Python runtime)
FROM python:3.11-slim

# 2️⃣ Set working directory inside container
WORKDIR /app

# 3️⃣ Copy requirements first (for caching)
COPY requirements.txt .

# 4️⃣ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy application code
COPY app.py .

# 6️⃣ Expose port (optional – for local testing)
EXPOSE 8080

# 7️⃣ Command to run your app
CMD ["python", "app.py"]
