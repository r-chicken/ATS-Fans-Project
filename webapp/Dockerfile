# Deploy target: Render (recommended, free) or Azure App Service for
# Containers (Linux), as an alternative for a work-provisioned Azure
# subscription. See webapp/README.md for the actual deploy steps.
FROM python:3.11-slim

# tesseract-ocr is the same system dependency the Colab notebook installs
# with `!apt-get install tesseract-ocr` - needed here too since this app
# OCRs each uploaded PDF's chart image the same way process_pdf always has.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (its own layer) so a code-only change doesn't
# force reinstalling sentence-transformers/torch on every rebuild.
COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir -r /app/webapp/requirements.txt

# Pre-download the sentence-embedding model at BUILD time, not on first
# request - avoids a slow (and network-dependent) first request in
# production, and means the container works even if outbound internet
# access is restricted at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ats_priority_checker (the shared pipeline code) plus this app itself.
# Build context is the REPO ROOT, not webapp/ - see the `docker build`
# command in README.md.
COPY ats_priority_checker /app/ats_priority_checker
COPY webapp /app/webapp

# The trained model bundle - see README.md for how it gets here before
# building. Not committed to git (webapp/.gitignore); copied in fresh
# each time you retrain and rebuild.
# (Already included via `COPY webapp /app/webapp` above if webapp/model/
# is populated at build time - called out separately here as the one
# piece of this image that changes independently of the app's code.)

EXPOSE 80
WORKDIR /app/webapp

# render-entrypoint.sh handles Render's "Secret Files" model-loading
# step (see webapp/README.md, "3. Deploy to Render") and is a harmless
# no-op everywhere else - it just execs gunicorn the same as before once
# that's done.
RUN chmod +x /app/webapp/render-entrypoint.sh
ENTRYPOINT ["/app/webapp/render-entrypoint.sh"]
