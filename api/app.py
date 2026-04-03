from fastapi import FastAPI,HTTPException
import uvicorn
from api.schema import DrugInput,field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from rdkit.Chem import Draw
import base64
from io import BytesIO
from src.utils import mol_structure,molecular_weight


app=FastAPI()

MODEL_PATH=r"D:\THIS PC\python_workspace\Drug_Discovery\models\drug_discovery_v1"
tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)
model=AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


@app.get('/')
def home():
    return{
        "message":"BBB classifier"
    }


@app.get('/health')
def health():
    return{
        "status":"OK"
    }

@app.post('/predict')
def predict(input:DrugInput):
    try:
        tokens=tokenizer(
            input.smiles,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        with torch.no_grad():
            outputs=model(**tokens)
            logits=outputs.logits
            prob=torch.sigmoid(logits[0][1]).item()

            mw=molecular_weight(input.smiles)
            structure_img=mol_structure(input.smiles)
        

            return{
                "smiles":input.smiles,
                "bbb_permeable":prob>=0.5,
                "label":"BBB+" if prob>=0.5 else "BBB-",
                'molecular_weight':mw,
                "molecule_structure":structure_img,
                "confidence":round(prob,4)
                

        }   
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

     