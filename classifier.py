print("Running classifier.py with ALL methods!")

import pandas as pd, json, torch, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset
from collections import Counter

def load_data(path):
    with open(path) as f: return pd.DataFrame.from_dict(json.load(f), orient="index")

def run_baseline_classifiers(df):
    Xq, yq = df["question"],     df["out-of-scope"]
    Xh, yh = df["human_answer"], df["invalid_HA"]

    Xq_tr,Xq_te,yq_tr,yq_te = train_test_split(Xq,yq,test_size=0.3,random_state=42)
    Xh_tr,Xh_te,yh_tr,yh_te = train_test_split(Xh,yh,test_size=0.3,random_state=42)

    vq, vha = TfidfVectorizer(stop_words="english"), TfidfVectorizer(stop_words="english")
    Xq_tr_tf,Xq_te_tf = vq.fit_transform(Xq_tr),  vq.transform(Xq_te)
    Xh_tr_tf,Xh_te_tf = vha.fit_transform(Xh_tr), vha.transform(Xh_te)

    lr_q  = LogisticRegression(max_iter=1000).fit(Xq_tr_tf,yq_tr)
    lr_ha = LogisticRegression(max_iter=1000).fit(Xh_tr_tf,yh_tr)

    yq_pred, yh_pred = lr_q.predict(Xq_te_tf), lr_ha.predict(Xh_te_tf)

    print("==== Baseline (LogReg) ====")
    print("Out‑of‑Scope Acc:", accuracy_score(yq_te,yq_pred))
    print(classification_report(yq_te,yq_pred,zero_division=0))
    print("Invalid_HA Acc:",  accuracy_score(yh_te,yh_pred))
    print(classification_report(yh_te,yh_pred,zero_division=0))

    return Xq_te.tolist(), yq_te.tolist(), Xh_te.tolist(), yh_te.tolist()

def run_zero_shot_with_roberta(Xq,yq,Xh,yh):
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")
    yq_pred = [clf(q,["in-scope","out-of-scope"])["labels"][0]=="out-of-scope" for q in Xq]
    yh_pred = [clf(a,["invalid","valid"])["labels"][0]=="invalid" for a in Xh]
    print("==== Zero‑Shot (BART) ====")
    print("Out‑of‑Scope Acc:",accuracy_score(yq,yq_pred)); print(classification_report(yq,yq_pred,zero_division=0))
    print("Invalid_HA  Acc:",accuracy_score(yh,yh_pred)); print(classification_report(yh,yh_pred,zero_division=0))

def run_fine_tuned_distilbert(df):
    print("\n==== Fine‑Tuned DistilBERT ====")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tok_q(ex):  return tok(ex["question"],     truncation=True,padding="max_length",max_length=128)
    def tok_ha(ex): return tok(ex["human_answer"], truncation=True,padding="max_length",max_length=128)

    # scope
    scope_df = df[["question","out-of-scope"]].rename(columns={"out-of-scope":"label"})
    scope_df["label"]=scope_df["label"].astype(int)
    ds_sc = Dataset.from_pandas(scope_df).train_test_split(test_size=0.3)
    ds_sc_tok = ds_sc.map(tok_q, batched=True)
    m_sc = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)
    m_sc.config.problem_type="single_label_classification"
    ta_sc = TrainingArguments(output_dir="distil_scope",evaluation_strategy="epoch",num_train_epochs=3,
                              per_device_train_batch_size=16,per_device_eval_batch_size=64,logging_strategy="no",save_strategy="no",seed=42)
    tr_sc = Trainer(model=m_sc,args=ta_sc,train_dataset=ds_sc_tok["train"],eval_dataset=ds_sc_tok["test"],
                    tokenizer=tok,compute_metrics=lambda p:{"accuracy":accuracy_score(p.label_ids,np.argmax(p.predictions,1)),
                                                             "report":classification_report(p.label_ids,np.argmax(p.predictions,1),zero_division=0)})
    tr_sc.train()
    print("DistilBERT Out‑of‑Scope:",tr_sc.predict(ds_sc_tok["test"]).metrics)

    # invalid
    inv_df = df[["human_answer","invalid_HA"]].rename(columns={"invalid_HA":"label"})
    inv_df["label"]=inv_df["label"].astype(int)
    ds_inv = Dataset.from_pandas(inv_df).train_test_split(test_size=0.3)
    ds_inv_tok = ds_inv.map(tok_ha, batched=True)
    m_inv = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)
    m_inv.config.problem_type="single_label_classification"
    ta_inv = TrainingArguments(output_dir="distil_invalid",evaluation_strategy="epoch",num_train_epochs=3,
                               per_device_train_batch_size=16,per_device_eval_batch_size=64,logging_strategy="no",save_strategy="no",seed=42)
    tr_inv = Trainer(model=m_inv,args=ta_inv,train_dataset=ds_inv_tok["train"],eval_dataset=ds_inv_tok["test"],
                     tokenizer=tok,compute_metrics=lambda p:{"accuracy":accuracy_score(p.label_ids,np.argmax(p.predictions,1)),
                                                              "report":classification_report(p.label_ids,np.argmax(p.predictions,1),zero_division=0)})
    tr_inv.train()
    print("DistilBERT Invalid_HA:",tr_inv.predict(ds_inv_tok["test"]).metrics)

def run_balanced_logreg(df):
    print("\n==== [LR‑BAL] Class‑weighted Logistic Regression ====")
    Xq,yq = df["question"],df["out-of-scope"]
    Xh,yh = df["human_answer"],df["invalid_HA"]
    Xq_tr,Xq_te,yq_tr,yq_te = train_test_split(Xq,yq,test_size=0.3,random_state=42)
    Xh_tr,Xh_te,yh_tr,yh_te = train_test_split(Xh,yh,test_size=0.3,random_state=42)

    vq,vh = TfidfVectorizer(stop_words="english"),TfidfVectorizer(stop_words="english")
    Xq_tr_tf,Xq_te_tf = vq.fit_transform(Xq_tr),vq.transform(Xq_te)
    Xh_tr_tf,Xh_te_tf = vh.fit_transform(Xh_tr),vh.transform(Xh_te)

    lr_q  = LogisticRegression(max_iter=1000,class_weight="balanced").fit(Xq_tr_tf,yq_tr)
    lr_ha = LogisticRegression(max_iter=1000,class_weight="balanced").fit(Xh_tr_tf,yh_tr)
    print("Out‑of‑Scope:",classification_report(yq_te,lr_q.predict(Xq_te_tf),zero_division=0))
    print("Invalid_HA :",classification_report(yh_te,lr_ha.predict(Xh_te_tf),zero_division=0))

class WeightedTrainer(Trainer):
    def __init__(self,class_weights,*args,**kw): super().__init__(*args,**kw); self.w=class_weights
    def compute_loss(self,model,inputs,return_outputs=False):
        labels=inputs.pop("labels"); out=model(**inputs); logits=out.logits
        loss=torch.nn.functional.cross_entropy(logits,labels,weight=self.w.to(model.device))
        return (loss,out) if return_outputs else loss

def run_weighted_bert(df):
    print("\n==== [BERT‑W] Weighted BERT‑base ====")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    def build(task_name,text_col,label_col):
        tmp=df[[text_col,label_col]].rename(columns={label_col:"label"}); tmp["label"]=tmp["label"].astype(int)
        ds=Dataset.from_pandas(tmp).train_test_split(test_size=0.3,seed=42)
        func=lambda ex:tok(ex[text_col],truncation=True,padding="max_length",max_length=128)
        return ds.map(func,batched=True)
    for task,text_col,label_col in [("Out‑of‑Scope","question","out-of-scope"),
                                    ("Invalid_HA","human_answer","invalid_HA")]:
        ds_tok=build(task,*[text_col,label_col])
        counts=Counter(ds_tok["train"]["label"]); total=sum(counts.values())
        w=torch.tensor([total/counts[i] for i in range(2)],dtype=torch.float)
        model=AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
        model.config.problem_type="single_label_classification"
        args=TrainingArguments(output_dir=f"bertw_{task}",evaluation_strategy="epoch",
                               num_train_epochs=4,per_device_train_batch_size=16,
                               per_device_eval_batch_size=64,logging_strategy="no",
                               save_strategy="no",seed=42)
        trainer=WeightedTrainer(w,model=model,args=args,
                                train_dataset=ds_tok["train"],eval_dataset=ds_tok["test"])
        trainer.train()
        m=trainer.predict(ds_tok["test"]).metrics
        print(f"{task}: Acc={m['test_accuracy']:.3f}")
        print(m["test_report"])

if __name__=="__main__":
    df = load_data("900_data.json")

    # original three
    Xq,yq,Xh,yh = run_baseline_classifiers(df)
    run_zero_shot_with_roberta(Xq,yq,Xh,yh)
    run_fine_tuned_distilbert(df)

    # NEW, higher‑accuracy additions
    run_balanced_logreg(df)    # LR‑BAL
    run_weighted_bert(df)      # BERT‑W
