# ATS Vibration Priority Checker - standalone scoring app

Upload a report PDF, get back whether the text and the Spectrum chart
agree with the priority the report states. This is deliberately separate
from the Colab notebook: it never runs OCR across a whole PDF folder or
retrains anything, it just loads the model Colab already trained and
scores whatever gets uploaded, the same fast per-report path the
notebook's own "Section 7: Scoring brand-new reports" cell uses.

I can't run the Azure deployment steps myself - I don't have access to
your Azure subscription from here - so this is a copy-pasteable guide for
you to run. If a command fails because of how your organization's Azure
is set up (permissions, naming policies, a required tag, etc.), that's
expected to vary - tell me the exact error and I can help adjust the
command, I just can't run it for you.

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

## 3. Deploy to Azure App Service

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
   server/Username/Password from step 2 (replace the placeholders):

   ```bash
   docker login <login-server> -u <username> -p <password>
   docker build -t <login-server>/ats-priority-checker:latest -f webapp/Dockerfile .
   docker push <login-server>/ats-priority-checker:latest
   ```

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

## 4. Updating later (new model, or code changes)

Whenever you retrain in Colab, or pull code updates into this repo, copy
the fresh model files into `webapp/model/` (step 1 above), then rebuild
and push:

```bash
docker build -t <image-tag-you-used-before> -f webapp/Dockerfile .
docker push <image-tag-you-used-before>
```

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
