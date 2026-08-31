# ATS Vibration Priority Checker - standalone scoring app

Upload a report PDF, get back whether the text and the Spectrum chart
agree with the priority the report states. This is deliberately separate
from the Colab notebook: it never runs OCR across a whole PDF folder or
retrains anything, it just loads the model Colab already trained and
scores whatever gets uploaded, the same fast per-report path the
notebook's own "Section 7: Scoring brand-new reports" cell uses.

I can't run any of these deployment steps myself - I don't have access to
your cloud accounts from here - so this is a copy-pasteable guide for you
to run. If a step fails in a way this doesn't cover, tell me the exact
error and I can help adjust it, I just can't run it for you.

**Which deployment section to use:** start with **3. Deploy to Render**
below - it's free with no credit card required and far fewer steps.
Azure App Service is documented after it as an alternative, for a
work-provisioned/IT-managed Azure subscription specifically (a personal
Azure free trial can't actually run this - see that section's notes).

## 1. Get the trained model out of Colab

After a training run finishes in the notebook (Section 6, "Save the
model"), download these two files from your Drive `OUT_DIR/model/`
folder:

- `priority_classifier.joblib`
- `priority_classifier.meta.json`

Put both directly into `webapp/model/` in this repo (same folder as this
README). They're git-ignored on purpose - you'll replace them every time
you retrain, and a trained model binary doesn't belong in commit history.

## 2. (Optional) Test it locally first

If you have Docker installed locally:

```bash
# from the REPO ROOT, not webapp/ - the Dockerfile needs both
# ats_priority_checker/ and webapp/ in its build context
docker build -t ats-priority-checker -f webapp/Dockerfile .
docker run -p 8080:80 ats-priority-checker
```

Then open `http://localhost:8080` and try uploading a report PDF.

Without Docker, you can also run it directly (needs `tesseract-ocr`
installed locally, same as any of this project's other pixel-reading
work):

```bash
cd webapp
pip install -r requirements.txt
python app.py   # http://localhost:8000
```

## 3. Deploy to Render (recommended - free, no credit card)

[Render](https://render.com) hosts a Docker image straight from your
GitHub repo, no separate registry or CLI needed - it's a much shorter
path than Azure. One caveat worth knowing up front: Render's free tier
gives 512 MB of RAM, and this app's embedding model can be memory-hungry
enough to bump into that. Worth trying first regardless since it's free
and quick to set up - if it crashes on startup or on the first upload,
that's the signal to fall back to Streamlit Community Cloud instead (ask
me and I'll adapt this app for that path, same idea, 1 GB free instead
of 512 MB).

1. **Sign up at [render.com](https://render.com)** using your GitHub
   account - no credit card anywhere in this flow.

2. **New -> Web Service**, and connect the GitHub repo this code lives
   in. Pick the branch you want deployed.

3. On the settings screen:
   - **Runtime**: Docker
   - **Dockerfile Path**: `webapp/Dockerfile`
   - **Docker Build Context Directory**: `.` (a single dot - the repo
     root, since the Dockerfile needs both `ats_priority_checker/` and
     `webapp/` alongside each other)
   - **Instance Type**: Free

   Click **Deploy Web Service**. It'll fail on this first deploy (no
   model loaded yet) - that's expected, continue to the next step.

4. **Add your trained model as Secret Files**, so it never has to go
   into git (same reasoning as `webapp/.gitignore`). In the new
   service's page, left sidebar -> **Environment** -> **Secret Files**
   -> **+ Add Secret File**, twice:

   - **Filename**: `priority_classifier.meta.json` -> **Contents**: open
     that file from your computer in any text editor, copy everything,
     paste it in.
   - **Filename**: `priority_classifier.joblib.b64` -> **Contents**: this
     one's a binary file, so it needs to be converted to text first.
     Open **PowerShell** on your computer (Start menu -> search
     "PowerShell") and run, adjusting the path to wherever your
     downloaded file actually is:
     ```powershell
     [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\priority_classifier.joblib")) | Set-Clipboard
     ```
     That copies a long block of text to your clipboard - paste it
     directly into the Contents field on Render.

   Click **Save Changes** - this triggers a fresh deploy that picks up
   both files automatically (the Dockerfile's entrypoint script decodes
   the `.b64` one back to a real file before starting the app).

5. Give it a few minutes. Once the deploy shows **Live**, open the URL
   shown at the top of the service page (something like
   `https://ats-priority-checker.onrender.com`) and try uploading a
   report PDF.

**Updating later:** retraining just means replacing the
`priority_classifier.joblib.b64` / `priority_classifier.meta.json`
Secret File contents the same way (Environment -> Secret Files -> edit
-> Save Changes) - no rebuild command to run yourself. A code change
just needs a normal `git push` to the branch Render is watching; it
redeploys automatically.

## Alternative: Deploy to Azure App Service

Use this instead of Render only if you have (or can get) a real,
work-provisioned Azure subscription - not a personal Azure free trial,
which is blocked from the compute quota this needs (see the quota
troubleshooting further down if you're unsure which kind you have).

Two ways to do this - same end result, pick whichever fits what you have
installed. **If you don't have (or can't install) the Azure CLI - e.g.
no admin rights on this machine - use Option B.** It needs nothing beyond
Docker Desktop (already installed) and a browser.

### Option A: Azure CLI

These are the standard steps for "custom container on Azure App Service"
- adjust names/region for your organization's conventions. Run from the
repo root.

```bash
# Log in (opens a browser) - use your work account
az login

# Pick names - RESOURCE_GROUP/LOCATION may already be dictated by your
# org's policies, check with whoever manages your Azure subscription
RESOURCE_GROUP="ats-priority-checker-rg"
LOCATION="eastus"
ACR_NAME="atspriority$RANDOM"        # must be globally unique, lowercase
PLAN_NAME="ats-priority-checker-plan"
APP_NAME="ats-priority-checker-$RANDOM"   # must be globally unique - this becomes your URL

# 1. Resource group (skip if your org already has one you should use)
az group create --name $RESOURCE_GROUP --location $LOCATION

# 2. Container registry to hold the built image
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# 3. Build and push the image
az acr login --name $ACR_NAME
docker build -t $ACR_NAME.azurecr.io/ats-priority-checker:latest -f webapp/Dockerfile .
docker push $ACR_NAME.azurecr.io/ats-priority-checker:latest

# 4. App Service plan (B1 is the smallest that reliably has enough memory
# for the embedding model - a free/F1 tier will likely fail to start)
az appservice plan create --resource-group $RESOURCE_GROUP --name $PLAN_NAME --is-linux --sku B1

# 5. The web app itself, pointed at the image you just pushed
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan $PLAN_NAME \
  --name $APP_NAME \
  --deployment-container-image-name $ACR_NAME.azurecr.io/ats-priority-checker:latest \
  --docker-registry-server-user $ACR_NAME \
  --docker-registry-server-password "$ACR_PASSWORD"

# 6. Make sure App Service knows which port the container listens on
az webapp config appsettings set --resource-group $RESOURCE_GROUP --name $APP_NAME --settings WEBSITES_PORT=80
```

Give it a couple of minutes to start (first boot pulls the image), then
open `https://$APP_NAME.azurewebsites.net` - print `$APP_NAME` if you
didn't note it down (`echo $APP_NAME`).

### Option B: Azure Portal (no CLI, no admin-requiring installs)

Everything below happens at **portal.azure.com** in your browser, plus
`docker build`/`docker push`/`docker login` in your terminal (Docker
Desktop, already installed - not Azure CLI). Sign into the Portal with
your work account first.

1. **Resource group.** Search "Resource groups" in the top search bar ->
   **+ Create**. Pick your Subscription, name it (e.g.
   `ats-priority-checker-rg`), pick a Region -> **Review + create** ->
   **Create**.

2. **Container registry.** Search "Container registries" -> **+ Create**.
   Resource group = the one you just made. Registry name = something
   globally unique, lowercase/numbers only (e.g. `atspriority1234`) -
   this becomes part of a URL, so the Portal will tell you immediately if
   it's taken. SKU = Basic -> **Review + create** -> **Create**.

   Once it's created, open it, go to **Settings -> Access keys** in the
   left menu, and toggle **Admin user** to Enabled. This page now shows a
   **Login server** (e.g. `atspriority1234.azurecr.io`), a **Username**,
   and a **Password** - copy all three somewhere, you need them next.

3. **Build and push the image**, from the repo root, using the Login
   server/Username/Password from step 2. Copy these three commands, then
   in your own copy **replace the whole word** `YOUR_LOGIN_SERVER` (etc.)
   with the real value - don't leave any `<` or `>` characters in there,
   those aren't part of the placeholder, and your terminal will read them
   as "redirect this to/from a file" and error out:

   ```bash
   docker login YOUR_LOGIN_SERVER -u YOUR_USERNAME -p YOUR_PASSWORD
   docker build -t YOUR_LOGIN_SERVER/ats-priority-checker:latest -f webapp/Dockerfile .
   docker push YOUR_LOGIN_SERVER/ats-priority-checker:latest
   ```

   For example, if your Login server is `atspriority1234.azurecr.io`,
   that first line should end up looking like
   `docker login atspriority1234.azurecr.io -u atspriority1234 -p (the password you copied, pasted in as-is)`
   - a plain word right after `-u`/`-p`/`login`, no angle brackets
   anywhere.

   **If `docker build` fails with something like "failed to connect to
   the docker API" / "virtualization support not detected":** Docker
   Desktop needs your machine's virtualization feature turned on to run
   at all, and on a company-managed laptop that's usually locked down by
   IT policy - not something you can fix yourself without admin rights.
   Skip local Docker entirely and use **Azure Cloud Shell** instead,
   which builds the image on Azure's servers, not your laptop:

   1. In the Portal, click the **`>_`** icon in the top toolbar to open
      Cloud Shell -> choose **Bash** (first time, accept the default
      storage it offers to create).
   2. On your computer, zip up your whole project folder (the one that
      already has `webapp/model/priority_classifier.joblib` and
      `.meta.json` in it).
   3. In Cloud Shell, click the **Upload/download files** icon and
      upload that zip.
   4. Then run:
      ```bash
      unzip YOUR_ZIP_NAME.zip -d ats-project
      cd ats-project
      az acr build --registry YOUR_REGISTRY_NAME --image ats-priority-checker:latest --file webapp/Dockerfile .
      ```
      (`YOUR_REGISTRY_NAME` is just the name part, e.g. `atspriority1234`
      - not the full `.azurecr.io` login server. If the unzip created a
      nested folder, `cd` into that instead.) Cloud Shell is already
      signed into your Azure account, so there's no login step, and this
      one command replaces `docker login`/`build`/`push` entirely -
      nothing runs on your laptop.

   Either way you get here, the image ends up in the same place - continue
   to step 4 below once it's pushed.

4. **App Service plan.** Search "App Service plans" -> **+ Create**.
   Resource group = the same one. Name it (e.g.
   `ats-priority-checker-plan`). Operating System = **Linux**. Region =
   same as before. Under Pricing plan, click **Explore pricing plans** (or
   "Change size") and pick **B1** (the free/F1 tier doesn't have enough
   memory for the embedding model and will fail to start) -> **Review +
   create** -> **Create**.

5. **The web app itself.** Search "App Services" -> **+ Create -> Web
   App**. Resource group = the same one. Name it - this becomes your URL
   (`<name>.azurewebsites.net`), so it needs to be globally unique too.
   **Publish = Container.** Operating System = Linux. Region = same as
   before. App Service Plan = the one from step 4.

   Move to the **Container** (or **Docker**) tab: Image Source = **Azure
   Container Registry**, then pick your registry, image
   (`ats-priority-checker`), and tag (`latest`) from the dropdowns - since
   you're signed into the same Azure account, this handles the
   registry credentials for you automatically, no separate
   username/password entry needed here.

   -> **Review + create** -> **Create**.

6. **Set the port.** Open the new Web App -> **Settings -> Configuration**
   in the left menu -> **Application settings** tab -> **+ New
   application setting**. Name = `WEBSITES_PORT`, Value = `80` -> **OK**
   -> **Save** at the top -> confirm.

Give it a couple of minutes to start (first boot pulls the image), then
open `https://<your-app-name>.azurewebsites.net`.

### Option C: GitHub Actions (no local Docker, no Azure CLI, no admin needed)

Use this if both A and B are blocked for you - e.g. Docker Desktop won't
start because virtualization is disabled by IT policy, *and* `az acr
build` fails with `TasksOperationsNotAllowed` (some organizations disable
ACR Tasks tenant-wide). This builds the image on GitHub's own servers
instead, using a workflow already included in this repo at
`.github/workflows/build-and-push-webapp.yml` - nothing runs on your
laptop or in your Azure subscription's build tooling at all.

1. **Regenerate your registry password first**, if it's ever been shared
   anywhere it shouldn't (pasted in chat, etc.) - you're about to store
   it as a GitHub secret, so start from a fresh one. Portal -> your
   registry -> **Settings -> Access keys** -> the regenerate button next
   to a password.

2. **Add three repository secrets.** In your GitHub repo:
   **Settings -> Secrets and variables -> Actions -> New repository
   secret**, and add each of these (name exactly as shown, value from
   your registry's Access keys page):
   - `ACR_LOGIN_SERVER` - e.g. `atspriority1234.azurecr.io`
   - `ACR_USERNAME` - e.g. `atspriority1234`
   - `ACR_PASSWORD` - the password from step 1

3. **Publish the model as a GitHub Release.** On GitHub: **Releases ->
   Draft a new release**. Give it any tag (e.g. `model-v1`), then drag
   both `priority_classifier.joblib` and `priority_classifier.meta.json`
   (from your local `webapp/model/` folder) into the "Attach binaries"
   box near the bottom -> **Publish release**. (This keeps the model out
   of the repo's regular commit history, same reasoning as
   `webapp/.gitignore` - a Release's attached files are separate
   storage, which is exactly why this workflow pulls from there instead
   of from the repo itself.)

4. **Run the workflow.** Go to the **Actions** tab -> **Build and push
   webapp image** in the left sidebar -> **Run workflow** button -> **Run
   workflow** again to confirm. Give it a couple of minutes and watch for
   a green checkmark - it downloads the model files from the Release you
   just published, builds the image, and pushes it to your registry.

Once that's green, continue to step 4 below (App Service plan) exactly
as written - same registry, same image, same tag, regardless of which
option (A/B/C) built it.

**Updating later:** publish a new Release (any new tag) with the fresh
model files attached, then re-run the workflow from the Actions tab - it
always grabs the most recent Release automatically.

## 4. Updating later (Azure path - new model, or code changes)

Whenever you retrain in Colab, or pull code updates into this repo, copy
the fresh model files into `webapp/model/` (step 1 above), then rebuild
and push:

```bash
docker build -t YOUR_LOGIN_SERVER/ats-priority-checker:latest -f webapp/Dockerfile .
docker push YOUR_LOGIN_SERVER/ats-priority-checker:latest
```

(Same `YOUR_LOGIN_SERVER` as step 3 above - whatever you used there, e.g.
`atspriority1234.azurecr.io`. No angle brackets.)

Then tell the Web App to pick up the new image - **CLI**:
`az webapp restart --resource-group $RESOURCE_GROUP --name $APP_NAME` -
or **Portal**: open the Web App -> **Overview** -> **Restart** button at
the top.

(Optional, for either path: on the Web App's **Deployment Center** page
you can turn on **Continuous Deployment**, so pushing a new `:latest`
image to the registry auto-redeploys without needing that manual
restart step at all.)

## What this app does and doesn't do

- Does: OCR + parse each uploaded PDF the same way `process_pdf` always
  has, read the Spectrum peak off the chart pixels, embed the
  Recommendations/Comments text with the same frozen sentence-embedding
  model training used, and run the loaded classifier - then show the same
  text/spectrum-vs-stated comparison as `priority_recommendation_table`.
- Doesn't: retrain, look at any report history (no escalation signal -
  that needs a whole equipment's dated history, which a single upload
  doesn't have), or write anything back to Drive/dataset.csv. Every
  upload is scored independently, fresh.
