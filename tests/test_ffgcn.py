from embiggen.edge_prediction.edge_prediction_ensmallen.ffgcn import FFGCN
from embiggen.embedders import FirstOrderLINEEnsmallen
from ensmallen.datasets.linqs import Cora, get_words_data

def test_ffgcn():
    """Testing that FFGCN works"""
    cora, _features = get_words_data(Cora())

    line = FirstOrderLINEEnsmallen()
    embedding = line.fit_transform(cora).get_all_node_embedding()[0]

    model = FFGCN(
        units=[100, 100, 100],
        number_of_steps_per_layer=100,
        pre_train=True,
        number_of_oversampling_neighbourhoods_per_node=10,
        threshold=3.0
    )

    model.fit(
        graph=cora,
        node_features=embedding
    )

    predictions = model.predict_proba(
        graph=cora,
        node_features=embedding
    )

    print(predictions)