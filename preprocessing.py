from collections import defaultdict
import sys

input_file = "com-orkut.ungraph.txt"
output_file = "orkut.graph"

print(f"Reading edges from {input_file}...")

graph = defaultdict(set)
max_node = 0
edge_count = 0

with open(input_file, "r") as f:
    for i, line in enumerate(f):
        if line.startswith("#"):
            continue
        try:
            u, v = map(int, line.strip().split())
            u += 1
            v += 1
            graph[u].add(v)
            graph[v].add(u)
            max_node = max(max_node, u, v)
            edge_count += 1
        except:
            continue
        if i % 1_000_000 == 0:
            print(f"Processed {i // 1_000_000}M lines...", flush=True)

print("Finished reading.")
print(f"Max node ID: {max_node}")
print(f"Total unique nodes: {len(graph)}")
print(f"Writing to METIS format in {output_file}...")

with open(output_file, "w") as f:
    total_edges = sum(len(neigh) for neigh in graph.values()) // 2
    f.write(f"{max_node} {total_edges}\n")
    for i in range(1, max_node + 1):
        neighbors = sorted(graph[i])
        f.write(" ".join(map(str, neighbors)) + "\n")

print("Done writing METIS file.")
