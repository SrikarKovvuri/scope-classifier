print("Training models on Group 0 (100%) and saving for testing on other groups")

import pandas as pd, json, torch, numpy as np, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset
from collections import Counter
import os


def load_data(path):
    with open(path) as f:
        return pd.DataFrame.from_dict(json.load(f), orient="index")


def train_and_save_logreg(df, save_dir="models"):
    """Train LR-BAL on 100% of data and save models"""
    print("\n==== [LR‑BAL] Training on 100% of Group 0 ====")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all data (no train/test split)
    # SKIP out-of-scope since you only care about invalid_HA
    Xh, yh = df["human_answer"], df["invalid_HA"]
    
    print(f"Training on {len(Xh)} human answers")
    print(f"Invalid_HA distribution: {yh.value_counts().to_dict()}")
    
    # Train vectorizer and model for invalid_HA only
    vh = TfidfVectorizer(stop_words="english")
    Xh_tf = vh.fit_transform(Xh)
    
    lr_ha = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ).fit(Xh_tf, yh)
    
    # Save models (only invalid_HA)
    model_objects = {
        "vh": vh, 
        "lr_ha": lr_ha
    }
    
    with open(f"{save_dir}/lr_models.pkl", "wb") as f:
        pickle.dump(model_objects, f)
    
    print(f"✅ Saved LR models to {save_dir}/lr_models.pkl")
    return model_objects


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=self.class_weights.to(model.device)
        )
        return (loss, outputs) if return_outputs else loss


def train_and_save_bert(df, epochs=4, save_dir="models"):
    """Train BERT-W on 100% of data and save models"""
    print("\n==== [BERT‑W] Training on 100% of Group 0 ====")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_objs = {}

    def make_full_ds(text_col, label_col):
        """Create dataset from 100% of data (no train/test split)"""
        tmp = df[[text_col, label_col]].rename(columns={label_col: "label"})
        tmp = tmp.astype({"label": int})
        ds = Dataset.from_pandas(tmp)
        return ds.map(
            lambda e: tok(
                e[text_col], truncation=True,
                padding="max_length", max_length=128
            ),
            batched=True
        )

    # Only train on invalid_HA (skip out-of-scope)
    for task, txt, lbl in [
        ("invalid_HA",   "human_answer", "invalid_HA")
    ]:
        print(f"\nTraining BERT for {task}...")
        
        ds_full = make_full_ds(txt, lbl)
        cnt = Counter(ds_full["label"])
        total = len(ds_full)
        class_wts = torch.tensor([total / cnt[i] for i in range(2)], dtype=torch.float)
        
        print(f"Training on {total} examples, label distribution: {dict(cnt)}")

        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        model.config.problem_type = "single_label_classification"

        args = TrainingArguments(
            output_dir=f"{save_dir}/bertw_{task}",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            logging_strategy="epoch",
            save_strategy="epoch",
            evaluation_strategy="no",  # No validation since we're using 100%
            seed=42
        )

        trainer = WeightedTrainer(
            class_wts,
            model=model,
            args=args,
            train_dataset=ds_full
        )
        trainer.train()
        
        # Save the trainer and tokenizer
        trainer.save_model(f"{save_dir}/bertw_{task}")
        tok.save_pretrained(f"{save_dir}/bertw_{task}")

        bert_objs[task] = {
            "model_path": f"{save_dir}/bertw_{task}",
            "text_col": txt,
            "label_col": lbl
        }
        
        print(f"✅ Saved BERT {task} model to {save_dir}/bertw_{task}")

    return bert_objs


if __name__ == "__main__":
    # Load Group 0 consensus data
    print("Loading Group 0 consensus data...")
    train_df = load_data("group0_consensus.json")
    
    print(f"Group 0 data shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    
    # Train and save both models on 100% of Group 0
    lr_models = train_and_save_logreg(train_df)
    bert_models = train_and_save_bert(train_df)
    
    print(f"\n🎉 All models trained and saved!")
    print(f"📁 Models saved in 'models/' directory")
    print(f"📊 Ready to test on Groups 1, 2, 3!")