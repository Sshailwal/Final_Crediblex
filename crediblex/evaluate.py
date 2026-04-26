import torch
import pandas as pd
import config
from transformers import AutoTokenizer, DebertaV2Tokenizer
from model import NewsTrustModel
import math
import os

# Suppress HuggingFace cache warnings for paths with spaces
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


def load_model_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    # Support both raw state_dict and dict containing 'model_state_dict'
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(k.startswith('backbone') or k in model.state_dict() for k in ckpt.keys()):
        # Might already be a state_dict
        state = ckpt
    else:
        # Try common key names
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    try:
        model.load_state_dict(state)
        print("Loaded checkpoint into model.")
    except Exception as e:
        print("Failed to load state_dict directly:", e)
        raise


def batch_tokenize(tokenizer, texts, max_len):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_len)


def evaluate(df, model, tokenizer, device, batch_size=8, max_len=config.MAX_LEN):
    model.eval()
    preds = {
        'bias': [],
        'factuality': [],
        'intent': [],
        'emotion': []
    }
    trues = {
        'bias': [],
        'factuality': [],
        'intent': [],
        'emotion': []
    }

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start:start+batch_size]
            texts = batch_df['text'].astype(str).tolist()
            tok = batch_tokenize(tokenizer, texts, max_len)
            input_ids = tok['input_ids'].to(device)
            attention_mask = tok['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)

            # Bias / Intent / Emotion -> class logits
            bias_logits = outputs['bias'].cpu()
            intent_logits = outputs['intent'].cpu()
            emotion_logits = outputs['emotion'].cpu()
            factuality = outputs['factuality'].cpu().squeeze()
            
            bias_preds = torch.argmax(bias_logits, dim=1).tolist()
            intent_preds = torch.argmax(intent_logits, dim=1).tolist()
            emotion_preds = torch.argmax(emotion_logits, dim=1).tolist()
            fact_preds = factuality.tolist()

            preds['bias'].extend(bias_preds)
            preds['intent'].extend(intent_preds)
            preds['emotion'].extend(emotion_preds)
            preds['factuality'].extend(fact_preds)

            trues['bias'].extend(batch_df['bias_label'].astype(int).tolist())
            trues['intent'].extend(batch_df['intent_label'].astype(int).tolist())
            trues['emotion'].extend(batch_df['emotion_label'].astype(int).tolist())
            trues['factuality'].extend(batch_df['fact_score'].astype(float).tolist())

    # Compute simple metrics
    def accuracy(y_true, y_pred):
        correct = sum(1 for a, b in zip(y_true, y_pred) if int(a) == int(b))
        return correct / len(y_true) if len(y_true) > 0 else 0.0

    def mse(y_true, y_pred):
        n = len(y_true)
        if n == 0:
            return 0.0
        return sum((float(a) - float(b)) ** 2 for a, b in zip(y_true, y_pred)) / n

    metrics = {
        'bias_acc': accuracy(trues['bias'], preds['bias']),
        'intent_acc': accuracy(trues['intent'], preds['intent']),
        'emotion_acc': accuracy(trues['emotion'], preds['emotion']),
        'factuality_mse': mse(trues['factuality'], preds['factuality'])
    }

    return preds, trues, metrics


if __name__ == '__main__':
    device = config.DEVICE
    print('Device:', device)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'training_data.csv')

    # Load sample data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f'Failed to read {csv_path}:', e)
        raise SystemExit(1)

    if df.shape[0] == 0:
        print('No rows in training_data.csv')
        raise SystemExit(1)

    # Use a manageable sample for quick evaluation
    sample_size = min(256, len(df))
    eval_df = df.head(sample_size).reset_index(drop=True)
    print(f'Evaluating on {len(eval_df)} rows (head of training_data.csv)')

    # Load tokenizer + model
    # Use AutoTokenizer to stay consistent with train.py and inference.py
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=False)
    model = NewsTrustModel(config.MODEL_NAME).to(device)

    model_path = os.path.join(script_dir, config.SAVE_PATH)
    try:
        load_model_checkpoint(model, model_path, device)
    except Exception as e:
        print('Error loading model checkpoint:', e)
        raise SystemExit(1)

    preds, trues, metrics = evaluate(eval_df, model, tokenizer, device, batch_size=config.BATCH_SIZE, max_len=config.MAX_LEN)

    print('\n=== Metrics ===')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'{k}: {v:.4f}')
        else:
            print(f'{k}:', v)

    # Show first 10 example predictions
    print('\n=== Examples (first 10) ===')
    for i in range(min(10, len(eval_df))):
        text = str(eval_df.loc[i, 'text'])
        print(f'-- Row {i} --')
        print('Text snippet:', (text[:320] + '...') if len(text) > 320 else text)
        print('True -> bias:', trues['bias'][i], 'fact:', trues['factuality'][i], 'intent:', trues['intent'][i], 'emotion:', trues['emotion'][i])
        print('Pred -> bias:', preds['bias'][i], 'fact:', round(preds['factuality'][i], 4), 'intent:', preds['intent'][i], 'emotion:', preds['emotion'][i])
        print()

    # Optionally save predictions to CSV
    out_df = eval_df.copy()
    out_df['pred_bias'] = preds['bias']
    out_df['pred_fact'] = preds['factuality']
    out_df['pred_intent'] = preds['intent']
    out_df['pred_emotion'] = preds['emotion']
    out_csv = os.path.join(script_dir, 'eval_predictions.csv')
    out_df.to_csv(out_csv, index=False)
    print(f'Saved predictions to {out_csv}')
