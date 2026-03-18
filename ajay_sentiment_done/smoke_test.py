import torch
import pandas as pd
import config
from transformers import AutoTokenizer
from model import NewsTrustModel
import train  # Import train module to access and set global TOKENIZER

print('Using Python:', __import__('sys').executable)
print('torch:', torch.__version__, 'cuda_available=', torch.cuda.is_available())

# Load tiny sample
df = pd.read_csv('training_data.csv').head(2)
if df.shape[0] == 0:
    raise SystemExit('No data in training_data.csv')

# Prepare tokenizer + batch
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
# Set the global TOKENIZER in train module for collate_fn to use
train.TOKENIZER = tokenizer

items = []
for _, row in df.iterrows():
    items.append({
        'text': row['text'],
        'bias_label': torch.tensor(int(row['bias_label']), dtype=torch.long),
        'fact_label': torch.tensor(float(row['fact_score']), dtype=torch.float),
        'intent_label': torch.tensor(int(row['intent_label']), dtype=torch.long),
        'emotion_label': torch.tensor(int(row['emotion_label']), dtype=torch.long)
    })

batch = train.collate_fn(items, max_len=config.MAX_LEN)

# Move to device
device = config.DEVICE
print('Device to use:', device)

model = NewsTrustModel(config.MODEL_NAME).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for k in ['input_ids','attention_mask']:
    batch[k] = batch[k].to(device)
for k in ['bias_label','fact_label','intent_label','emotion_label']:
    batch[k] = batch[k].to(device)

# Do one training step
loss_fn_class = torch.nn.CrossEntropyLoss()
loss_fn_reg = torch.nn.MSELoss()

if device == 'cuda':
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss_bias = loss_fn_class(outputs['bias'], batch['bias_label'])
        loss_fact = loss_fn_reg(outputs['factuality'].squeeze(), batch['fact_label'])
        loss_intent = loss_fn_class(outputs['intent'], batch['intent_label'])
        loss_emotion = loss_fn_class(outputs['emotion'], batch['emotion_label'])
        loss = loss_bias + loss_fact + loss_intent + loss_emotion
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    outputs = model(batch['input_ids'], batch['attention_mask'])
    loss_bias = loss_fn_class(outputs['bias'], batch['bias_label'])
    loss_fact = loss_fn_reg(outputs['factuality'].squeeze(), batch['fact_label'])
    loss_intent = loss_fn_class(outputs['intent'], batch['intent_label'])
    loss_emotion = loss_fn_class(outputs['emotion'], batch['emotion_label'])
    loss = loss_bias + loss_fact + loss_intent + loss_emotion
    loss.backward()
    optimizer.step()

print('Smoke test completed successfully. Loss:', loss.item())
if device == 'cuda':
    print('GPU memory allocated (GB):', torch.cuda.memory_allocated()/1e9)
