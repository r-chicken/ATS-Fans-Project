# ATS Vibration Priority Checker - Streamlit Community Cloud version

Same app as `webapp/` (see that folder's README for what it does and
doesn't do) - this version exists because Streamlit Community Cloud
doesn't run a Dockerfile the way Render/Azure do, it runs a Streamlit
app directly, so the web page itself is written differently here. The
actual scoring logic (reading the PDF, checking the chart, running the
classifier) is identical either way.

Use this if Render's free tier ran out of memory for you (a crash with
exit code 137, or Render emailing about a "server failure" / "Killed" in
the logs) - Streamlit Community Cloud's free tier gives 1 GB of RAM
instead of Render's 512 MB.

## 1. Create the app

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub - no credit card anywhere in this flow.
2. **Create app** -> pick this repository and the branch you want
   deployed.
3. **Main file path**: `streamlit_app/app.py`
4. Click **Deploy**. It'll fail on this first run (no model loaded yet)
   - that's expected, continue to the next step.

## 2. Add your trained model as a Secret

Streamlit Community Cloud's Secrets are a plain-text box (TOML format),
so the model files go in there instead of git - same reasoning as
`webapp/.gitignore` (a trained model binary doesn't belong in commit
history, and you'll replace it every time you retrain).

On the app's page: **Settings (⋮ menu, or the gear icon) -> Secrets**,
and paste in this template, filling in the two placeholders:

```toml
[model]
joblib_b64 = "PASTE_BASE64_TEXT_HERE"
meta_json = """
PASTE_RAW_CONTENTS_OF_priority_classifier.meta.json_HERE
"""
```

- **`meta_json`**: open `priority_classifier.meta.json` (from your
  Colab run's Drive output) in any text editor, copy everything, and
  paste it between the `"""` lines exactly as-is.
- **`joblib_b64`**: `priority_classifier.joblib` is a binary file, so it
  needs converting to text first. On Windows, open **PowerShell** (Start
  menu -> search "PowerShell"), right-click the `.joblib` file in File
  Explorer -> **Copy as path**, then run:
  ```powershell
  [Convert]::ToBase64String([IO.File]::ReadAllBytes(
  ```
  paste the copied path right after that `(`, then finish the line:
  ```powershell
  )) | Set-Clipboard
  ```
  That copies a long text block to your clipboard - paste it in place
  of `PASTE_BASE64_TEXT_HERE` (keep the surrounding quotes already in
  the template).

Click **Save** - the app restarts automatically and picks up both
values.

## 3. Open it

Back on the app's page, click **App URL** (something like
`https://<your-app-name>.streamlit.app`) and try uploading a report PDF.

## Updating later

- **New model after retraining**: repeat step 2 with the fresh files -
  Settings -> Secrets -> replace both values -> Save. No rebuild command
  to run yourself.
- **Code changes**: a normal `git push` to the branch this app is
  watching - it redeploys automatically.
