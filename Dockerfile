FROM python:3.10-slim

# Install system dependencies (cv2 dependencies if needed, git for diffusers)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml .
# COPY uv.lock .  # Uncomment if lock file exists

# Install dependencies
# We use --system to install into the system python environment since we are in a container
RUN uv pip install --system -r pyproject.toml

# Copy source code
COPY src/ ./src/
COPY .env.example .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
