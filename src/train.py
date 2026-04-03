import pandas as pd
from src.preprocess import preprocess
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,Trainer,TrainingArguments,AutoModelForSequenceClassification
from torch.utils.data import  Dataset 
from torch.utils.data import DataLoader
import mlflow
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from scipy.special import softmax
from peft import LoraConfig,get_peft_model,TaskType





def tokenize_data(tokenizer,X_train,X_test):

    train_tokens=tokenizer(
        list(X_train),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
)

    test_tokens=tokenizer(
        list(X_test),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
)
    return train_tokens,test_tokens

class BBBDataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=labels.values

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        item={key:torch.tensor(val[index]) for key,val in self.encodings.items()}
        item['labels']=torch.tensor(self.labels[index])

        return item
    

def create_dataset(train_tokens,test_tokens,y_train,y_test):
    train_dataset=BBBDataset(train_tokens,y_train)
    test_dataset=BBBDataset(test_tokens,y_test)
    return train_dataset,test_dataset

def load_dataset(train_dataset,test_dataset):
        train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
        test_loader=DataLoader(test_dataset,batch_size=16)
        return train_loader,test_loader

class WeightedTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False, **kwargs):
        labels=inputs.pop('labels')
        outputs=model(**inputs)
        logits=outputs.get("logits")
        loss_fct=torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss=loss_fct(logits,labels)
        return (loss,outputs) if return_outputs else loss
    
model=AutoModelForSequenceClassification.from_pretrained(
    "DeepChem/ChemBERTa-10M-MTR",
    num_labels=2
    )

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("BBB Classification")


def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds=logits.argmax(axis=1)
    probs=softmax(logits,axis=1)[:,1]

    return{"accuracy":accuracy_score(labels,preds),
            "f1":f1_score(labels,preds),
           "roc_auc":roc_auc_score(labels,probs)
          }

def finetuning(train_dataset,test_dataset):
    training_args=TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        eval_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='eval_roc_auc',
        fp16=True,
    )
    with mlflow.start_run() as run:
        trainer=WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics=trainer.evaluate()
        mlflow.log_metric("eval_loss",metrics["eval_loss"])
        trainer.save_model("./models/drug_discovery_v1")
        tokenizer.save_pretrained("./models/drug_discovery_v1")
        mlflow.log_artifacts("./models/drug_discovery_v1",artifact_path="model")
        
        

def lora_finetune(train_dataset,test_dataset):

    model=AutoModelForSequenceClassification.from_pretrained(
    "DeepChem/ChemBERTa-10M-MTR",
    num_labels=2
    )
    
    
    lora_config=LoraConfig(

        r=16,
        lora_alpha=32,
        target_modules=["query","value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    model=get_peft_model(model,lora_config)
    

    training_args=TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        eval_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='eval_roc_auc',
        fp16=True,
        )
    
    with mlflow.start_run():

        mlflow.set_tag("training_type","lora")
        trainer=WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        #trainer.train()
        #metrics=trainer.evaluate()
        #mlflow.log_metric("eval_loss",metrics["eval_loss"])


    
if __name__=="__main__":
    

    df = pd.read_csv("data/B3DB_classification.csv", sep="\t")
    df = preprocess()

    X = df["SMILES"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MTR")

    train_tokens, test_tokens = tokenize_data(tokenizer, X_train, X_test)

    train_dataset, test_dataset = create_dataset(
        train_tokens, test_tokens, y_train, y_test
    )

    # class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    finetuning(train_dataset, test_dataset)

    lora_finetune(train_dataset,test_dataset)



    
    
    
    