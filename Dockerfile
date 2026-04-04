# Hugging Face Spaces — Docker + Streamlit
# Uses a slim Python base to keep image size manageable.
FROM python:3.11-slim

# HF Spaces runs the container as a non-root user (uid 1000).
# Create the user and set ownership up front.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install OS-level dependencies first (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies using the slim requirements file
COPY requirements-spaces.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY --chown=appuser:appuser . .

# Create directories that are gitignored but needed at runtime
RUN mkdir -p outputs/models outputs/reports data/processed data/raw \
    && chown -R appuser:appuser outputs data

USER appuser

# HF Spaces exposes port 7860
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Use the app.py shim at the project root as the Streamlit entry point
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
