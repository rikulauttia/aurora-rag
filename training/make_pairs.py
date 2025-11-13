import json, random

# Hyvin pieni demo: query -> relevant doc index
PAIRS = [
  ("IT-ura- ja rekrymessu Turussa", 0),
  ("72 tunnin AI-hackathon Turussa", 1),
  ("Missä opiskella ML ja AI Suomessa", 2),
  ("Aalto yliopisto koneoppiminen maisteri", 2),
  ("rekrymessut opiskelijoille Turku", 0),
  ("yrityshaasteet hackathonissa", 1),
]

# Ladataan meta ja tehdään negatiivisia
meta = json.load(open("inference/meta.json","r",encoding="utf-8"))
all_idx = list(range(len(meta)))

train = []
for q, pos_i in PAIRS:
    pos_text = meta[pos_i]["text"]
    # yksi negatiivinen satunnaisesti
    neg_i = random.choice([i for i in all_idx if i != pos_i])
    neg_text = meta[neg_i]["text"]
    train.append({"query": q, "doc": pos_text, "label": 1})
    train.append({"query": q, "doc": neg_text, "label": 0})

json.dump(train, open("training/train_pairs.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("Wrote training/train_pairs.json")
