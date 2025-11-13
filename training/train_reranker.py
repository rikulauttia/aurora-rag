import json, os, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from tqdm import tqdm

BASE = os.environ.get("BASE_CE","cross-encoder/ms-marco-MiniLM-L-6-v2")
OUT = os.environ.get("OUT","models/aurora-reranker-lora")
LR = float(os.environ.get("LR","2e-5"))
EPOCHS = int(os.environ.get("EPOCHS","2"))
BSZ = int(os.environ.get("BSZ","8"))

data = json.load(open("training/train_pairs.json","r",encoding="utf-8"))
texts1 = [d["query"] for d in data]
texts2 = [d["doc"] for d in data]
labels = [d["label"] for d in data]
ds = Dataset.from_dict({"text1":texts1,"text2":texts2,"label":labels})

tok = AutoTokenizer.from_pretrained(BASE)
def collate(batch):
    t1 = [b["text1"] for b in batch]
    t2 = [b["text2"] for b in batch]
    y  = torch.tensor([b["label"] for b in batch], dtype=torch.float)
    enc = tok(t1, t2, padding=True, truncation=True, return_tensors="pt", max_length=256)
    return enc, y

dl = DataLoader(ds, batch_size=BSZ, shuffle=True, collate_fn=collate)

model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query","value","key","dense"], lora_dropout=0.05, bias="none", task_type="SEQ_CLS")
model = get_peft_model(model, peft_cfg)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

total_steps = len(dl)*EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

model.train()
for ep in range(EPOCHS):
    pbar = tqdm(dl, desc=f"Epoch {ep+1}/{EPOCHS}")
    for (enc, y) in pbar:
        enc = {k:v.to(device) for k,v in enc.items()}
        y = y.to(device).unsqueeze(-1)
        out = model(**enc)
        loss = torch.nn.functional.mse_loss(out.logits, y)
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
        pbar.set_postfix({"loss": float(loss.item())})

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("Saved LoRA reranker to", OUT)
