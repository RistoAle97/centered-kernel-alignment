from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast

from ckatorch import CKA


def tokenize(sentences_to_tokenize: dict[str, str]):
    return bert_tokenizer(
        sentences_to_tokenize["sentence1"],
        sentences_to_tokenize["sentence2"],
        truncation=True,
        add_special_tokens=True,
        padding="longest",
        max_length=bert_tokenizer.model_max_length,
        return_tensors="pt",
    ).to("cuda:0")


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
    dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=True)

    # Observable_layers
    layers = [f"encoder.layer.{i}.output.dense" for i in range(bert_config.num_hidden_layers)]

    # Build the CKA objects, one for confronting a model with itself and the other for comparing two different models
    cka_same_model = CKA(
        first_model=first_model,
        second_model=first_model,
        layers=layers,
        first_name="Bert",
        device="cuda:0",
    )
    cka_different_models = CKA(
        first_model=first_model,
        second_model=second_model,
        layers=layers,
        first_name="Bert_0",
        second_name="Bert_1",
        device="cuda:0",
    )

    # Compute the CKA values for both scenarios
    cka_matrix_same = cka_same_model(dataloader, epochs=1)
    cka_matrix_different = cka_different_models(dataloader, epochs=1)

    # Plot the CKA values
    plot_parameters = {
        # "show_ticks_labels": True,
        # "short_tick_labels_splits": 4,
        "use_tight_layout": True,
        "show_half_heatmap": True,
    }
    cka_same_model.plot_cka(
        cka_matrix=cka_matrix_same,
        title=f"Model {cka_same_model.first_model_info.name} compared with itself",
        **plot_parameters,
    )
    cka_different_models.plot_cka(cka_matrix=cka_matrix_different, **plot_parameters)
