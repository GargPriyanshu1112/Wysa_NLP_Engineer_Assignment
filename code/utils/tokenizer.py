from transformers import AutoTokenizer


class CustomTokenizer:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(self, docs, column):
        return self.tokenizer(docs[column])

    def __call__(self, docs, column):
        return docs.map(lambda batch: self.tokenize(batch, column), batched=True)
