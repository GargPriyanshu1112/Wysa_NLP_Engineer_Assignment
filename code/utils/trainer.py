from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments


class CustomTrainer:
    def __init__(
        self,
        train_dataset,
        tokenizer,
        model_id,
        num_classes,
        output_dir,
        num_train_epochs,
        train_batch_size=64,
        save_total_limit=1,
        eval_strategy="no",
        save_strategy="epoch",
    ):
        self.dataset = train_dataset
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_classes
        )
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=train_batch_size,
            save_total_limit=save_total_limit,
            evaluation_strategy=eval_strategy,
            save_strategy=save_strategy,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
        )

    def __call__(self):
        self.trainer.train()
