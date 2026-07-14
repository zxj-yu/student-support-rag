# Deployment & Upgrade Guide

This guide goes with four files: `Dockerfile`, `.github/workflows/deploy.yml`, `agent_layer.py`, `evaluate.py`.
Follow them in order — each completed step gives you one more concrete line to add to your resume/README.

---

## Step 1: Deploy the project to GCP Cloud Run (roughly 1-2 hours)

### Why Cloud Run
- Generous free tier (2 million requests/month free) — a personal project basically costs nothing
- Pay-per-request billing; with `min-instances=0`, you pay nothing when there's no traffic
- Deploying is just "push a container" — no need to deal with the complexity of K8s, well suited for a portfolio project

### Steps

1. **Sign up for a GCP account**, create a new project, and note down your `PROJECT_ID`

2. **Install the gcloud CLI locally**, then log in:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Enable the required APIs:**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   gcloud services enable secretmanager.googleapis.com
   ```

4. **Store your Qdrant API key in Secret Manager** (don't hardcode it in your code — this is basic security practice and interviewers will ask about it):
   ```bash
   echo -n "your Qdrant API Key" | gcloud secrets create qdrant-api-key --data-file=-
   ```

5. **Do a manual deploy first to confirm everything works:**
   ```bash
   gcloud run deploy student-support-rag \
     --source . \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi
   ```
   This will give you a `https://xxx.run.app` URL — this is the "production environment URL" you can put on your resume.

6. **Set up GitHub Actions for automated deployment** (using `.github/workflows/deploy.yml`):
   - In GCP, create a Service Account and grant it Cloud Run Admin permissions
   - Generate a JSON key for the Service Account
   - In your GitHub repo's Settings → Secrets, add:
     - `GCP_PROJECT_ID`
     - `GCP_SA_KEY` (the full contents of the JSON key)
     - `QDRANT_URL`
   - After this, every push to main will automatically run tests, build the image, and deploy — a complete CI/CD loop

### Add monitoring (takes 15 minutes, but is a strong plus)
Cloud Run comes with basic monitoring built in — just check the Cloud Run page in the GCP Console for:
- Request latency (p50/p95/p99)
- Error rate
- Instance count

Take a few screenshots of these charts and add them to your README — this is the most direct proof that you understand how to monitor a production system.

### Track your costs (write this into your README)
After a week of running, take a screenshot of the GCP Billing page and write something like:
> "This project handles an average of X requests/day, with monthly cloud costs of roughly $Y, of which Z% comes from vector search"
A quantified description like this is far more convincing than a vague "I deployed it to the cloud."

---

## Step 2: Add agent capability (using `agent_layer.py`)

1. Drop `agent_layer.py` into your project's root directory
2. Wire up the `search_course_documents` and `check_assignment_status` functions
   to your existing Qdrant retrieval logic (and a real data source, if you have one)
3. In your FastAPI routes in `main.py`, replace the place where you directly call
   RAG retrieval with a call to `run_agent(question, retriever)`
4. Test a few questions that require multi-step reasoning to confirm the model
   really does decide on its own which tool to call

**How to talk about this change in an interview:**
"The original system followed a fixed pipeline: retrieve, then generate. I added
a layer that lets the model decide which tool to use and whether it needs
multiple rounds of calls, so the system can now handle compound questions it
couldn't handle before."

---

## Step 3: Add an evaluation framework (using `evaluate.py`)

1. Replace `TEST_CASES` with questions your project can actually answer (start with 10-15)
2. Run `python evaluate.py` once and check the average score
3. Optionally, wire this evaluation script into GitHub Actions too (extra credit, but very much worth it):
   run the evaluation automatically on every push and treat the score as a quality gate

---

## Priority guide

If time is limited, do these in order of cost-effectiveness (highest to lowest):
1. **Cloud deployment** (even just "manually deployed successfully, with a working URL") — highest payoff
2. **Evaluation framework** (write 10 test cases, get an average score) — second highest, few candidates do this
3. **Agent capability** (tool-use layer) — most technically substantial, but also the most time-consuming

Once you're done, update your README to include:
- Production URL
- An architecture description (can just be plain text: FastAPI -> Agent decision layer -> Qdrant retrieval / tool calls -> generation)
- Evaluation results (average score, test set size)
- Cost data
