# 1. Base Image
# We use a specific, lightweight "slim" image to minimize vulnerabilities and size
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Optimize caching by copying ONLY requirements first
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the source code and models
COPY . .

# 6. Run the application
CMD ["python", "bot/main.py"]