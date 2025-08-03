import json

from safetensors.torch import load_file

from ckatorch.plot import plot_cka


if __name__ == "__main__":
    # Load the info file
    with open("cka.json") as info_file:
        model_info = json.load(info_file)

    # Load the CKA matrix
    cka_matrix = load_file(model_info["cka_matrix"]["path"])["cka"]

    # Plot the CKA values
    plot_cka(
        cka_matrix=cka_matrix,
        first_layers=model_info["first_model"]["layers"],
        second_layers=model_info["second_model"]["layers"],
        save_path=".",
        show_ticks_labels=True,
        short_tick_labels_splits=2,
        use_tight_layout=True,
        show_half_heatmap=True,
    )
