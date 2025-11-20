# Dockerfile for MintEDGE
FROM python:3.11-slim

# Install SUMO and system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        sumo sumo-tools sumo-doc \
    && rm -rf /var/lib/apt/lists/*

# SUMO in Debian/Ubuntu typically lives here
ENV SUMO_HOME=/usr/share/sumo

# Create working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the repo
COPY . .

# Default command (you can override on docker run)
CMD ["python", "MintEDGE.py", "--help"]
