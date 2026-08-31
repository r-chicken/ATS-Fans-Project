#!/bin/sh
# Render-specific startup step - safe to run anywhere else too (it's a
# no-op if these files aren't present, e.g. on Azure or a local run).
#
# The trained model still never goes into git (see webapp/.gitignore).
# On Render there's no build-time step to copy it in from elsewhere, so
# instead it's pasted into Render's "Secret Files" dashboard field and
# shows up here at runtime under /etc/secrets/. The .joblib is binary,
# so it's stored base64-encoded (Render's secret file editor is a plain
# text box) and decoded back to real bytes here before the app starts.
set -e

mkdir -p /app/webapp/model

if [ -f /etc/secrets/priority_classifier.joblib.b64 ]; then
  base64 -d /etc/secrets/priority_classifier.joblib.b64 > /app/webapp/model/priority_classifier.joblib
fi

if [ -f /etc/secrets/priority_classifier.meta.json ]; then
  cp /etc/secrets/priority_classifier.meta.json /app/webapp/model/priority_classifier.meta.json
fi

# Render tells the container which port to listen on via $PORT; fall
# back to 80 (what the Dockerfile EXPOSEs) for Azure/local runs where
# that variable isn't set.
exec gunicorn --bind "0.0.0.0:${PORT:-80}" --timeout 180 --workers 2 app:app
