from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
import base64
from io import BytesIO
import requests


def molecular_weight(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return Descriptors.MolWt(mol)


def mol_structure(smiles):
    mol=Chem.MolFromSmiles(smiles)
    
    img=Draw.MolToImage(mol)
    buffer=BytesIO()
    img.save(buffer,format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()

def mol_name(smiles):

    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/synonyms/JSON"
        res = requests.get(url).json()

        name = res['InformationList']['Information'][0]['Synonym'][0]
        return name

    except:
        return "Unknown"
