print("Evaluating trained models on other groups (100% testing)")

import pandas as pd, json, torch, numpy as np, pickle
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import os


def load_data(path):
    with open(path) as f:
        return pd.DataFrame.from_dict(json.load(f), orient="index")


def load_lr_models(model_path="models/lr_models.pkl"):
    """Load saved LR models"""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate_lr_on_group(group_df, lr_models, group_name):
    """Evaluate LR models on a group"""
    print(f"\n==== LR-BAL Evaluation on {group_name} ====")
    
    # Only test invalid_HA (since that's all we trained)
    Xh_tf = lr_models["vh"].transform(group_df["human_answer"])
    yh = group_df["invalid_HA"].astype(int)
    yh_pred = lr_models["lr_ha"].predict(Xh_tf)
    
    print("Invalid_HA Results:")
    print(f"Accuracy: {accuracy_score(yh, yh_pred):.3f}")
    print(classification_report(yh, yh_pred, zero_division=0))
    
    return {
        "invalid_ha_acc": accuracy_score(yh, yh_pred),
        "invalid_ha_report": classification_report(yh, yh_pred, output_dict=True, zero_division=0)
    }


def evaluate_bert_on_group(group_df, group_name, models_dir="models"):
    """Evaluate BERT models on a group"""
    print(f"\n==== BERT-W Evaluation on {group_name} ====")
    
    results = {}
    
    # Only evaluate invalid_HA (since that's all we trained)
    for task, text_col, label_col in [
        ("invalid_HA", "human_answer", "invalid_HA")
    ]:
        print(f"\n-- BERT {task} --")
        
        model_path = f"{models_dir}/bertw_{task}"
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Prepare data
        tmp = group_df[[text_col, label_col]].rename(columns={label_col: "label"})
        tmp = tmp.astype({"label": int})
        ds = Dataset.from_pandas(tmp)
        ds_tok = ds.map(
            lambda e: tokenizer(e[text_col], truncation=True, padding="max_length", max_length=128),
            batched=True
        )
        
        # Make predictions
        model.eval()
        predictions = []
        true_labels = ds_tok["label"]
        
        with torch.no_grad():
            for i in range(len(ds_tok)):
                inputs = {k: torch.tensor([v]) for k, v in ds_tok[i].items() if k != "label"}
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)
        
        # Calculate metrics
        acc = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
        
        print(f"Accuracy: {acc:.3f}")
        print(f"Macro-F1: {report['macro avg']['f1-score']:.3f}")
        print(classification_report(true_labels, predictions, zero_division=0))
        
        results[task] = {
            "accuracy": acc,
            "macro_f1": report['macro avg']['f1-score'],
            "report": report
        }
    
    return results


def evaluate_single_group(group_file, group_name):
    """Evaluate both LR and BERT on a single group"""
    print(f"\n{'='*50}")
    print(f"EVALUATING {group_name.upper()}")
    print(f"{'='*50}")
    
    # Load group data
    group_df = load_data(group_file)
    print(f"Group size: {len(group_df)}")
    print(f"Invalid_HA distribution: {group_df['invalid_HA'].value_counts().to_dict()}")
    
    # Load and evaluate LR models
    lr_models = load_lr_models()
    lr_results = evaluate_lr_on_group(group_df, lr_models, group_name)
    
    # Evaluate BERT models  
    bert_results = evaluate_bert_on_group(group_df, group_name)
    
    # Summary
    print(f"\n📊 SUMMARY for {group_name}:")
    print(f"LR Invalid_HA Accuracy: {lr_results['invalid_ha_acc']:.3f}")
    print(f"BERT Invalid_HA Accuracy: {bert_results['invalid_HA']['accuracy']:.3f}")
    print(f"BERT Invalid_HA Macro-F1: {bert_results['invalid_HA']['macro_f1']:.3f}")
    
    return {
        "group": group_name,
        "lr_results": lr_results,
        "bert_results": bert_results
    }


def evaluate_all_groups():
    """Evaluate on all test groups"""
    # Update these file paths when you process other groups
    test_groups = {
        "Group 1": "group3_consensus.json",

    }
    
    all_results = []
    
    for group_name, file_path in test_groups.items():
        if os.path.exists(file_path):
            results = evaluate_single_group(file_path, group_name)
            all_results.append(results)
        else:
            print(f"⚠️  {file_path} not found. Process {group_name} first.")
    
    return all_results


if __name__ == "__main__":
    # Check if models exist
    if not os.path.exists("models/lr_models.pkl"):
        print("❌ Models not found! Run the training script first.")
        exit(1)
    
    print("🚀 Starting evaluation on test groups...")
    
    # For now, let's just set up the framework
    # You'll run this after processing groups 1, 2, 3
    print("📝 Ready to evaluate! Process groups 1, 2, 3 first, then run:")
    print("   python evaluate_other_groups.py")
    
    # Uncomment this when you have the other group files:
    all_results = evaluate_all_groups()