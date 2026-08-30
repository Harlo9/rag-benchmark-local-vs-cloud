# STEP 10 / 10: API container image.
#
# Goal: make the service runnable on any machine with Docker, without a Python
# setup. Only the code lives in the image; data and models stay outside.

FROM python:3.12-slim

# Unbuffered output so logs appear immediately, and no .pyc files to write.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies are installed before the code is copied. Docker caches each layer,
# so editing a source file no longer triggers a full reinstall of the packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# Running as a non-root user: a container that does not need root privileges
# should not have them.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# 0.0.0.0 rather than localhost: inside a container, binding to localhost would
# make the service unreachable from the host.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]