import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import torch as th
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from absl import app, flags, logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import pytorch_lightning as pl
import sh


flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_boolean('debug', False, '')

FLAGS = flags.FLAGS


sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 64
label2id = {id:id+1 for id in range(-1, 3, 1)}
id2label = {v:k for k, v in label2id.items()} # reverse dict

class BertSentClassification(pl.LightningModule):
    def __init__(self, dataset, hidden_sz=200, output_sz=4, dropout_prob=0.2):
        super().__init__()
        self.dataset = dataset

        #Load pre-trained model
        self.bert_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased"
        )
        self.dropout = nn.Dropout(dropout_prob)
        # Add Multi layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(768, hidden_sz), #768 is the size of BERT output
            nn.ReLU(),
            nn.Linear(hidden_sz, output_sz)
        )

        # define metrics
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_f1 = pl.metrics.FBeta(
            num_classes=output_sz,
            beta=1,
            average="macro"
          )

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.bert_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # BERT vectors corresponding to the [CLS] token
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.mlp(pooled_output)

        return logits

    def configure_optimizers(self):
      #Define the optimizer: Stocastic Gradient Descent
      return th.optim.SGD(self.parameters(), lr=5e-3)

    def training_step(self, batch, batch_idx):
        # training_step will hold processing corresponding each training step
        # the epoch loop and batch training loop abstracted away by Pytorch Lighting
        input_ids, attention_mask, labels = batch

        logits = self(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        th.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        loss = F.cross_entropy(logits, labels)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True
        )

        return loss


    def validation_step(self, batch, batch_idx):
      # implementation corresponding to processing of validation data
      input_ids, attention_mask, labels = batch

      logits = self(
          input_ids,
          attention_mask = attention_mask,
          labels=labels
      )
      self.log(
          "validation_accuracy",
          self.valid_acc(logits, labels),
          on_epoch=True,
          prog_bar=True,
          logger=True
      )
      self.log(
            "validation_f1",
            self.valid_f1(logits, labels),
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

    def train_dataloader(self):
        # dataloader corresponding to training data
        train_sampler = RandomSampler(self.dataset["train"])

        return DataLoader(
            dataset=self.dataset["train"],
            sampler=train_sampler,
            batch_size=64
        )


    def val_dataloader(self):
        # dataloader corresponding to validation data
        val_sampler = SequentialSampler(self.dataset["val"])

        return DataLoader(
            dataset=self.dataset["val"],
            sampler=val_sampler,
            batch_size=64
        )

def convert_examples_to_features(tweets, labels):
    input_ids = [
        bert_tokenizer.encode(tweet, add_special_tokens=True) for tweet in tweets
    ]
    input_ids = pad_sequences(
        input_ids,
        maxlen=MAX_LEN,
        dtype="long",
        value=bert_tokenizer.pad_token_id,
        padding="post",
        truncating="post"
    )
    input_ids = th.tensor(input_ids)
    attention_masks = th.tensor([[int(tok > 0) for tok in tweet] for tweet in input_ids])
    labels = th.tensor([label2id[label] for label in labels])

    return TensorDataset(input_ids, attention_masks, labels)


def main(_):
    df = pd.read_csv("twitter_sentiment_data.csv")
    dataset = convert_examples_to_features(df.message, list(df.sentiment))
    train_data, val_data, train_label, val_labels = train_test_split(
        dataset,
        list(df.sentiment),
        random_state=1234,
        test_size=0.2
    )
    dataset = {"train": train_data, "val": val_data}
    model = BertSentClassification(dataset=dataset)
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='twitter', version=0),
    )
    trainer.fit(model)

    trainer.save_checkpoint('model_checkpoint.ckpt')







if __name__ == '__main__':
    app.run(main)