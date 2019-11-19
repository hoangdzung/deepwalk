deepwalk --format npy --input ~/graphsage_pytorch/data/cora/cora_matrix.npy \
--max-memory-data-size 0 --number-walks 80 --representation-size 128 --walk-length 40 --window-size 10 \
--workers 1 --output example_graphs/cora.embeddings
