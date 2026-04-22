**Live render deploymemt link:** https://chemberta-neurobarrier-classifier.onrender.com/

This is a web app I built to predict Blood-Brain Barrier (BBB) permeability for drug compounds. You just feed it a SMILES string, and it tells you if the compound is likely to cross the BBB or not.

## What is this solving?

In drug discovery, the Blood-Brain Barrier (BBB) is a massive hurdle. It's a highly selective border of cells that protects the brain from toxins in the blood. 
- If you're designing a drug for a neurological condition (like Alzheimer's or depression), your drug **must** cross the BBB to be effective.
- If you're designing a drug for the rest of the body (like heart or liver medication), you generally **want to avoid** crossing the BBB to prevent unwanted neurological side effects.

Testing whether a new compound can cross the BBB usually requires slow and expensive lab assays or animal testing. This project attempts to solve that problem by predicting BBB permeability instantly using machine learning, allowing researchers to filter compounds computationally very early in the drug design phase.

## About the Model

Under the hood, this uses a fine-tuned Hugging Face transformer model based on **ChemBERTa**. 

ChemBERTa is effectively the chemical equivalent of a language model. But instead of English sentences, it was pre-trained on millions of **SMILES strings** (which are text-based representations of chemical molecules). By fine-tuning this pre-trained model on a specific dataset of known BBB-permeable and non-permeable compounds (using PEFT techniques), the model has learned the underlying structural and chemical rules that dictate whether a molecule can slip through the blood-brain barrier. 

Because we have a FastAPI backend, the model inference is exposed via a clean REST API, and there's a simple, vanilla HTML/JS frontend that visualizes the molecules and calculates some basic stats like molecular weight using RDKit.

## Model Performance & Access

The model was evaluated on an unseen test dataset to ensure it generalizes well. Based on the final training results, it achieved the following metrics:
- **ROC AUC:** 0.914
- **F1 Score:** 0.871
- **Accuracy:** 83.3%

You can view or use the fine-tuned model directly from here:
👉 **[akshat3260/chemberta-bbb-classifier](https://huggingface.co/akshat3260/chemberta-bbb-classifier)
**

## Tech Stack
- **Backend:** FastAPI, Uvicorn
- **ML:** PyTorch, Transformers (ChemBERTa), Scikit-learn, PEFT
- **Chemistry:** RDKit (for 2D structures and molecular weights)
- **Frontend:** Vanilla HTML/CSS/JS
- **Dockerfile**
- **Model versioning:** DVC
- **Deployment**: Render
- **CI-CD:** GitHub Actions

## Project Layout
- `api/` - The FastAPI backend and data schemas
- `src/frontend/` - The HTML, CSS, and JS for the web interface
- `src/` - Scripts for preprocessing data and training the ML model
- `models/` - Where the saved model weights live (`drug_discovery_v1`)
- `data/` & `results/` - Datasets and training metrics

## How to run it

The easiest way to get this up and running is with Docker, especially since installing RDKit and PyTorch dependencies can sometimes be a hassle.

### Using Docker
```bash
docker build -t bbb-predictor .
docker run -p 8000:8000 bbb-predictor
```
Then just open `http://localhost:8000` in your browser.

### Running locally without Docker

If you have Python 3.11 ready to go, you can run it directly:

1. Clone the repo and navigate in:
   ```bash
   git clone <your-repo-link>
   cd Drug_Discovery
   ```

2. Install the requirements. I'd definitely recommend using a virtual environment (`python -m venv .venv`).
   ```bash
   pip install -r requirements.txt
   ```
   *Tip: If you're just running the app and not training the model, you probably want to install the CPU-only version of PyTorch first to save a ton of space.*

3. Start the server:
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

## Training the model

If you want to tweak the ML side of things, take a look at `src/train.py` and `src/preprocess.py`. The setup takes SMILES sequences, tokenizes them, and trains a sequence classification model to output the final permeability probability.

