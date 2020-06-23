from typing import Dict

space = {
    "p": (0.1, 10.0),  # (float)
    "q": (0.1, 10.0),  # (float)
    "num_walks": (1, 20),  # (int) This should be dependant on the graph size.
    "walk_length": (32, 256),  # (int)
    "embedding_size": (8, 512),  # (int)
    "context_window": (1, 5),  # (int)
    "num_epochs": (1, 4)  # (int)
}

config = dict(
    paths=dict(
        pos_train="path/to/my/pos_train",
        pos_valid="path/to/my/pos_valid",
        pos_test="path/to/my/pos_test",
        neg_train="path/to/my/neg_train",
        neg_valid="path/to/my/neg_valid",
        neg_tes="path/to/my/neg_tes"
    ),
    w2v_model="skipgram"
)


def custom_loss(config: Dict, reporter):
    pos_train, pos_valid, pos_test, neg_train, neg_valid, neg_test = read_graphs(
        **config["paths"]
    )
    walks = get_random_walks(
        pos_train,
        config["p"],
        config["q"],
        config["num_walks"],
        config["walk_length"]
    )

    reporter(my_custom_loss=x**2)
