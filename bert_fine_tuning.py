import pandas as pd
import numpy as np
import datasets
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments
from transformers import Trainer
import torch

# ------- pretrained modelのダウンロードとセットアップ
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
sc_model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=3)
sc_model = sc_model.to(device)

# https://dev.classmethod.jp/articles/huggingface-jp-text-classification/
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

# ------- 杏仁豆腐、シュー、プリンのデータセットを用意

tsukurepo_df = pd.read_csv('tsukurepo_df.csv', encoding='ms932', sep=',',skiprows=0)
tsukurepo_df.sample(frac=1)
tsukurepo_texts = tsukurepo_df['tsukurepo'].values.tolist()
labels = tsukurepo_df['keyword'].values
uniq_l = np.unique(labels)
label_dic = {w:i for i,w in enumerate(uniq_l)}
label_dic_inv = {i:w for i,w in enumerate(uniq_l)}
#label_ids = np.array([label_dic[w] for w in labels])


text_data=[]
for l,txt in zip(labels,tsukurepo_texts):
    #print(txt)
    txt = txt.translate(str.maketrans({"\n":"", "\t":"", "\r":"", "\u3000":""}))
    text_data.append([txt,label_dic[l]])


#label_df = pd.DataFrame(label_ids.reshape(-1,1),columns=['label'])
tsukurepo = pd.DataFrame(text_data,columns=['text','label'])
#tsukurepo = pd.concat([text_df,label_df],axis=1)
#tsukurepo=tsukurepo.rename(columns = {'tsukurepo':'text'})
train_idx = int(len(tsukurepo)*0.75)
train_df = tsukurepo.iloc[:train_idx,:]
test_df = tsukurepo.iloc[train_idx:,:]

# ----------------- データセットをBERTの入力データ形式に変換
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


train_data = datasets.Dataset.from_pandas(train_df[['text', 'label']])
train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
train_data.set_format("torch", columns=["input_ids", "label"])
print('train_data',train_data)
input()
eval_data = datasets.Dataset.from_pandas(test_df[['text', 'label']])
eval_data = eval_data.map(tokenize, batched=True, batch_size=len(eval_data))
eval_data.set_format("torch", columns=["input_ids", "label"])




def compute_metrics(result):
    labels = result.label_ids
    preds = result.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }

# data, data/logs などはcolaboratoryで自動的に作成される
training_args = TrainingArguments(output_dir = "./data",logging_dir = "./data/logs",num_train_epochs =10,per_device_train_batch_size = 5,
                                  per_device_eval_batch_size = 32,
                                  warmup_steps=500,weight_decay=0.001,evaluation_strategy = "steps")

trainer = Trainer(
    model = sc_model,  # 使用するモデルを指定
    args = training_args,  # TrainingArgumentsの設定
    compute_metrics = compute_metrics,  # 評価用の関数
    train_dataset = train_data,  # 訓練用のデータ
    eval_dataset = eval_data  # 評価用のデータ
)

trainer.train()
print(trainer.evaluate())

model_path = "./data/"


sc_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)