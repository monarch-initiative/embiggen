from time import time
from humanize import naturaldelta
from embiggen import Graph, GraphFactory
import compress_json

start = time()
factory = GraphFactory(default_directed=True)
graph = factory.read_csv(
    "tests/data/first_walk_test_edges.tsv",
    "tests/data/first_walk_test_nodes.tsv",
    
)
graph.random_walk(number=10, length=80)
delta = time() - start

response = {
    "required_time": delta,
    "human_time": naturaldelta(delta)
}

print(response)

compress_json.dump(response, "time_required.json")