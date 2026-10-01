# CAD-COVID model service (result contract v1). Build from the repository root:
#   docker build -f service/Dockerfile -t cad-service .
# The U-Net weights (Git LFS) must be pulled first: `git lfs pull`. A pointer file instead of the
# weights is detected at start-up and the service reports it on /health (it never falls back).
FROM python:3.10-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY service/requirements.lock .
# pyradiomics 3.0.1 imports numpy in its setup.py: build it against the pinned numpy, without
# build isolation, then collect every wheel of the lock file (pyradiomics is found already built)
# so the runtime stage needs no compiler.
RUN pip install --no-cache-dir numpy==1.26.4 \
 && pip wheel --no-cache-dir --no-build-isolation --wheel-dir /wheels pyradiomics==3.0.1 \
 && pip wheel --no-cache-dir --wheel-dir /wheels --find-links /wheels -r requirements.lock

FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES="" \
    UPLOAD_FOLDER=/app/uploads RESULTS_FOLDER=/app/results PORT=5557
COPY service/requirements.lock /tmp/requirements.lock
# The wheels are mounted from the build stage, not copied: a COPY layer stays in the image even
# after deleting the files (measured: 4.2 GB image instead of 3.2 GB).
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements.lock
WORKDIR /app
COPY modules/inference.py modules/inference.py
COPY ["Analysis Dashboard/params.yaml", "Analysis Dashboard/extracted_features_Covid.csv", "Analysis Dashboard/extracted_features_normal.csv", "Analysis Dashboard/"]
# U-Net weights (~372 MB) are mounted from host at runtime or auto-downloaded if missing
RUN mkdir -p "ML/UNET Training"
COPY service service
EXPOSE 5557
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '5557'), timeout=4).status == 200 else 1)"
# exec: uvicorn becomes PID 1 and receives SIGTERM (docker stop), instead of being killed after 10 s.
CMD ["sh", "-c", "exec uvicorn service.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
