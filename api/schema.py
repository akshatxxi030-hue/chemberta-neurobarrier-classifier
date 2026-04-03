from pydantic import BaseModel,field_validator
from rdkit import Chem

class DrugInput(BaseModel):
    smiles:str

    @field_validator("smiles")
    def validate_smile(cls,v):

        if not isinstance(v,str):
            raise ValueError("SMILES must be a non-empty string")
        
        v=v.strip()

        if not v:
            raise ValueError("SMILES cannot be empty")
        
        mol=Chem.MolFromSmiles(v)
        if mol is None:
            raise ValueError("Invalid SMILES sequence")
        return v