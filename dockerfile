# Base image ndogo
FROM python:3.11-alpine

# Weka working directory
WORKDIR /app

# Install dependencies (kama unazo kwenye requirements.txt)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Run the app
CMD ["python", "main.py"]
