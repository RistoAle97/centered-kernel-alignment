from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast

from cka import CKA


def tokenize(sentences_to_tokenize: dict[str, str]):
    return bert_tokenizer(
        sentences_to_tokenize["sentence1"],
        sentences_to_tokenize["sentence2"],
        truncation=True,
        add_special_tokens=True,
        padding="longest",
        max_length=bert_tokenizer.model_max_length,
        return_tensors="pt",
    )


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("nyu-mll/glue", "mrpc", split="test", verification_mode="no_checks")

    # Load the tokenizer and the model from the hub
    bert_tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    bert_config = BertModel.from_pretrained("google-bert/bert-base-uncased").config
    first_model = BertModel(config=bert_config)
    second_model = BertModel(config=bert_config)

    # Define the dataloader
    dataset.set_transform(tokenize, ["sentence1", "sentence2"])
    dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=True, drop_last=True)

    # Observable_layers
    layers = [f"encoder.layer.{i}.output.dense" for i in range(bert_config.num_hidden_layers)]

    # Compute CKA
    cka_matrix = CKA(
        first_model=first_model,
        second_model=second_model,
        layers=layers,
    )
