# HuggingFace 日本語BERTのfine tuning 

[bert_fine_tuning.ipynb](bert_fine_tuning.ipynb)colaboratoryで実行する。ローカルで実行する際は、関連モジュールのバージョンの整合性が取れないことがあるので注意

``` python

# ------- pretrained modelのダウンロードとセットアップ
#cuda:GPUとプログラムを橋渡しするAPI GPUがセットアップされていればそれを使い、セットアップされていなければCPUを使うという宣言
# device：仮想的なコンピュータというようなイメージ
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
sc_model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=3)
# ダウンロードしたpre-trained modelを仮想コンピュータ空間に置く処理
sc_model = sc_model.to(device)

# https://dev.classmethod.jp/articles/huggingface-jp-text-classification/

# pre-trainedの時に使ったデータセットの辞書のようなものが入っている
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

# ------- 杏仁豆腐、シュー、プリンのデータセットを用意

tsukurepo_df = pd.read_csv('tsukurepo_df.csv', encoding='ms932', sep=',',skiprows=0)
tsukurepo_df.sample(frac=1)
tsukurepo_texts = tsukurepo_df['tsukurepo'].values.tolist()
labels = tsukurepo_df['keyword'].values
uniq_l = np.unique(labels)
label_dic = {w:i for i,w in enumerate(uniq_l)}
label_dic_inv = {i:w for i,w in enumerate(uniq_l)}


# ツクレポのテキストから改行コードや空白など余計な記号を除外
text_data=[]
for l,txt in zip(labels,tsukurepo_texts):
    #print(txt)
    txt = txt.translate(str.maketrans({"\n":"", "\t":"", "\r":"", "\u3000":""}))
    text_data.append([txt,label_dic[l]])



tsukurepo = pd.DataFrame(text_data,columns=['text','label'])

# 以下は何をやっているだろうか？
train_idx = int(len(tsukurepo)*0.75)
train_df = tsukurepo.iloc[:train_idx,:]
test_df = tsukurepo.iloc[train_idx:,:]

# ----------------- データセットをBERTの入力データ形式に変換
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

# dataframeで準備した訓練データ、テストデータをBERTの入力形式に直している。詳細は省略
train_data = datasets.Dataset.from_pandas(train_df[['text', 'label']])
train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
train_data.set_format("torch", columns=["input_ids", "label"])
print('train_data',train_data)

eval_data = datasets.Dataset.from_pandas(test_df[['text', 'label']])
eval_data = eval_data.map(tokenize, batched=True, batch_size=len(eval_data))
eval_data.set_format("torch", columns=["input_ids", "label"])


# ---------------  学習(fine tuning)と識別性能の評価

# HuggingFaceが提供するfine tuning用のライブラリ Trainer, TrainingArguments を使ってコーディングをシンプルにしている

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

# --------------  fine tuning済モデルの保存
sc_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

```