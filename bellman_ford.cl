__kernel void bellman_ford(
    __global const int* edgeStart,
    __global const int* edgeEnd,
    __global const int* edgeWeights,
    __global float* distances,
    __global int* updated,
    const int numEdges
) {
    int tid = get_global_id(0);
    if (tid < numEdges) {
        int u = edgeStart[tid];
        int v = edgeEnd[tid];
        int weight = edgeWeights[tid];
        
        // Relaxation step: update distance if a shorter path is found
        if (distances[u] != FLT_MAX && distances[u] + weight < distances[v]) {
            distances[v] = distances[u] + weight;
            atomic_min(&updated[0], 1); // Mark that an update has occurred
        }
    }
}

