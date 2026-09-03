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

Note for anyone touching this later: `tesseract-ocr` (the system package
this app needs for OCR, same as `webapp/Dockerfile` installs) is
declared in a `packages.txt` at the **repo root**, not inside this
folder - Streamlit Community Cloud only reads that file from the root,
even though `requirements.txt` is allowed to sit next to `app.py` like
it does here. Putting it in `streamlit_app/` instead is silently
ignored, no error, it just quietly never installs tesseract - if OCR
starts failing with "tesseract is not installed or it's not in your
PATH" again, check that `packages.txt` is still at the root first.

## 1. Create the app

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub - no credit card anywhere in this flow.
2. **Create app** -> pick this repository and the branch you want
   deployed.
3. **Main file path**: `streamlit_app/app.py`
4. Click **Deploy**. It'll fail on this first run (no model loaded yet)
   - that's expected, continue to the next step.

## 2. Add your trained models as Secrets

There's a single upload box - each uploaded report is automatically
sorted into a Fans, Pumps, or Blowers group (from the equipment
description text already parsed out of the report, e.g. "...Exhaust
Fan" vs. "...Hot Oil Pump" vs. "...Belt Blower") and scored against
that group's own trained model. That means there are up to three model
bundles to add, under three different Secrets keys.

Streamlit Community Cloud's Secrets are a plain-text box (TOML format),
so the model files go in there instead of git - same reasoning as
`webapp/.gitignore` (a trained model binary doesn't belong in commit
history, and you'll replace it every time you retrain).

On the app's page: **Settings (⋮ menu, or the gear icon) -> Secrets**,
and paste in this template, filling in the placeholders (fans' files
under `[model_fans]`, pumps' under `[model_pumps]`, blowers' under
`[model_blowers]`):

```toml
[model_fans]
joblib_b64 = "PASTE_BASE64_TEXT_HERE"
meta_json = """
PASTE_RAW_CONTENTS_OF_priority_classifier.meta.json_HERE
"""

[model_pumps]
joblib_b64 = "PASTE_BASE64_TEXT_HERE"
meta_json = """
PASTE_RAW_CONTENTS_OF_priority_classifier.meta.json_HERE
"""

[model_blowers]
joblib_b64 = "PASTE_BASE64_TEXT_HERE"
meta_json = """
PASTE_RAW_CONTENTS_OF_priority_classifier.meta.json_HERE
"""
```

For each section:

- **`meta_json`**: open that model's `priority_classifier.meta.json`
  (from its own Colab run's Drive output) in any text editor, copy
  everything, and paste it between the `"""` lines exactly as-is.
- **`joblib_b64`**: `priority_classifier.joblib` is a binary file, so it
  needs converting to text first. On Windows, open **PowerShell** (Start
  menu -> search "PowerShell"), right-click that model's `.joblib` file
  in File Explorer -> **Copy as path**, then run:
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

Click **Save** - the app restarts automatically and picks up whichever
sections you filled in. You don't need all three ready at once - fill
in whichever models you have and add the rest later; reports of a kind
with no model configured yet just come back with a note saying so,
while the kinds that ARE configured keep scoring normally.

## 3. Open it

Back on the app's page, click **App URL** (something like
`https://<your-app-name>.streamlit.app`) and try uploading a report PDF.

## Updating later

- **New model after retraining**: repeat step 2 with the fresh files,
  for whichever of `[model_fans]` / `[model_pumps]` / `[model_blowers]`
  changed - Settings -> Secrets -> replace that section's two values
  -> Save. No rebuild command to run yourself.
- **Code changes**: a normal `git push` to the branch this app is
  watching - it redeploys automatically.
