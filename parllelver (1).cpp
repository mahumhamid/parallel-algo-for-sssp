#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <algorithm>

using namespace std;

const int MAX_VERTICES = 4000000;  // Maximum number of vertices 
const int MAX_EDGES = 250000000;   // Maximum number of edges

struct Edge {
    int src, dest, weight;
};

// for 1d mapping
struct NodeMapping {
    int* nodeIdToIndex;
    int* indexToNodeId;
    int nextIndex;
    
    NodeMapping() {
        nodeIdToIndex = new int[MAX_VERTICES * 2]; 
        indexToNodeId = new int[MAX_VERTICES];
        nextIndex = 0;
        
        // Initialize with -1 to indicate not assigned
        for (int i = 0; i < MAX_VERTICES * 2; i++) {
            nodeIdToIndex[i] = -1;
        }
    }
    
    ~NodeMapping() {
        delete[] nodeIdToIndex;
        delete[] indexToNodeId;
    }
    
    int getIndex(int nodeId) {
        int hash = nodeId % (MAX_VERTICES * 2);  //hash to map nodes
        if (hash < 0) hash += (MAX_VERTICES * 2);
        
        while (true) {  //linear probing, for collision check
            if (nodeIdToIndex[hash] == -1) {
                nodeIdToIndex[hash] = nextIndex;
                indexToNodeId[nextIndex] = nodeId;
                nextIndex++;
                return nodeIdToIndex[hash];
            }
            
            if (nodeIdToIndex[hash] >= 0 && indexToNodeId[nodeIdToIndex[hash]] == nodeId) {
                return nodeIdToIndex[hash];
            }
            
            hash = (hash + 1) % (MAX_VERTICES * 2);
        }
    }
    
    int getNodeId(int index) const {
        if (index >= 0 && index < nextIndex) {
            return indexToNodeId[index];
        }
        return -1;
    }
    
    int getVertexCount() {
        return nextIndex;
    }
};

class ParallelGraph {
private:
    vector<Edge> edges;
    vector<Edge> localEdges;  // Edges assigned to this process
    int E;    // Current number of edges
    int V;    // Current number of vertices
    NodeMapping nodeMap;
    int rank;  // MPI rank
    int size;  // Number of MPI processes

public:
    ParallelGraph(int mpi_rank, int mpi_size) {
        E = 0;
        V = 0;
        rank = mpi_rank;
        size = mpi_size;
    }
    
    void addEdge(int src, int dest, int weight = 1) {
        int srcIndex = nodeMap.getIndex(src);
        int destIndex = nodeMap.getIndex(dest);
        
        V = nodeMap.getVertexCount();
        
        Edge e1 = {srcIndex, destIndex, weight};
        Edge e2 = {destIndex, srcIndex, weight};
        
        edges.push_back(e1);
        edges.push_back(e2);
        E += 2;
    }
    
    bool loadFromFile(const string& filename) {
        if (rank == 0) {
            cout << "Process " << rank << " loading graph file..." << endl;
        }
        
        ifstream file(filename.c_str());
        if (!file.is_open()) {
            if (rank == 0) {
                cerr << "Error: Unable to open file " << filename << endl;
            }
            return false;
        }
        
        string line;
        // Skip comment lines starting with '#'
        while (getline(file, line)) {
            if (line.empty() || line[0] != '#') {
                break;
            }
        }
        
        int lineCount = 0;
        int src, dest;
        
        // Process the first non-comment line
        if (!line.empty() && line[0] != '#') {
            istringstream iss(line);
            if (iss >> src >> dest) {
                addEdge(src, dest);
                lineCount++;
            }
        }
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;  // Skip empty lines or comments
            }
            
            istringstream iss(line);
            if (iss >> src >> dest) {
                addEdge(src, dest);
                lineCount++;
            }
            
            // Print progress every million edges (only from rank 0)
            if (rank == 0 && lineCount % 1000000 == 0) {
                cout << "Processed " << lineCount << " edges...\n";
            }
        }
        
        file.close();
        
        // Broadcast the total number of vertices to all processes
        V = nodeMap.getVertexCount();
        MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Distribute the edges among processes
        distributeEdges();
        
        if (rank == 0) {
            cout << "\nGraph loaded successfully.\n";
            cout << "Number of vertices: " << V << "\n";
            cout << "Number of undirected edges: " << E/2 << "\n";
            cout << "Edges distributed across " << size << " processes.\n";
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Process " << rank << " has " << localEdges.size() << " edges" << endl;
        
        return true;
    }
    
    void distributeEdges() {
        // Calculate how many edges each process should handle
        int edgesPerProcess = edges.size() / size;
        int remainder = edges.size() % size;
        
        int startIdx = rank * edgesPerProcess + min(rank, remainder);
        int endIdx = (rank + 1) * edgesPerProcess + min(rank + 1, remainder);
        
        // Assign local edges to this process
        localEdges.assign(edges.begin() + startIdx, edges.begin() + endIdx);
    }
    
    // Parallel Bellman-Ford algorithm
    int* parallelBellmanFord(int sourceNodeId) {
        double start_time = MPI_Wtime();
        
        int sourceIndex = -1;
        
        // Find the index for the source node ID
        for (int i = 0; i < V; i++) {
            if (nodeMap.getNodeId(i) == sourceNodeId) {
                sourceIndex = i;
                break;
            }
        }
        
        // Broadcast source index (root finds it, others receive it)
        MPI_Bcast(&sourceIndex, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (sourceIndex == -1) {
            if (rank == 0) {
                cerr << "Error: Source node " << sourceNodeId << " not found in the graph.\n";
            }
            return NULL;
        }
        
        // Initialize distance arrays
        int* dist = new int[V];
        int* global_dist = new int[V];
        
        for (int i = 0; i < V; i++) {
            dist[i] = INT_MAX;
            global_dist[i] = INT_MAX;
        }
        
        // Set source distance to 0
        if (rank == 0) {
            global_dist[sourceIndex] = 0;
        }
        
        // Broadcast initial distances
        MPI_Bcast(global_dist, V, MPI_INT, 0, MPI_COMM_WORLD);
        memcpy(dist, global_dist, V * sizeof(int));
        
        bool globalChange = true;
        int iteration = 0;
        
        // Relax edges |V| - 1 times
        while (globalChange && iteration < V - 1) {
            iteration++;
            bool localChange = false;
            
            // Each process relaxes its own subset of edges
            for (size_t j = 0; j < localEdges.size(); j++) {
                int u = localEdges[j].src;
                int v = localEdges[j].dest;
                int weight = localEdges[j].weight;
                
                if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    localChange = true;
                }
            }
            
            // Gather all distance arrays to process 0
            MPI_Allreduce(dist, global_dist, V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            
            // Check if any distance was updated
            int localUpdated = localChange ? 1 : 0;
            int globalUpdated = 0;
            
            MPI_Allreduce(&localUpdated, &globalUpdated, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            globalChange = (globalUpdated != 0);
            
            // Update local distances with the minimum values from all processes
            memcpy(dist, global_dist, V * sizeof(int));
            
            // Print progress (only from rank 0)
            if (rank == 0 && iteration % max(1, (V - 1) / 10) == 0) {
                cout << "Completed " << iteration << " of " << (V - 1) << " iterations (" 
                     << (100.0 * iteration / (V - 1)) << "%)\n";
            }
            
            // Early convergence check
            if (!globalChange) {
                if (rank == 0) {
                    cout << "Early convergence at iteration " << iteration << " of " << (V - 1) << endl;
                }
                break;
            }
        }
        
        // Check for negative weight cycles
        bool hasNegativeCycle = false;
        
        for (size_t j = 0; j < localEdges.size(); j++) {
            int u = localEdges[j].src;
            int v = localEdges[j].dest;
            int weight = localEdges[j].weight;
            
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                hasNegativeCycle = true;
                break;
            }
        }
        
        // Check if any process detected a negative cycle
        int localNegativeCycle = hasNegativeCycle ? 1 : 0;
        int globalNegativeCycle = 0;
        
        MPI_Allreduce(&localNegativeCycle, &globalNegativeCycle, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        
        if (globalNegativeCycle) {
            if (rank == 0) {
                cerr << "Graph contains negative weight cycle.\n";
            }
            delete[] dist;
            delete[] global_dist;
            return NULL;
        }
        
        // Only keep global_dist in the root process, others can free it
        if (rank != 0) {
            delete[] global_dist;
            delete[] dist;
            return NULL;
        } else {
            double end_time = MPI_Wtime();
            cout << "Parallel Bellman-Ford completed in " << end_time - start_time << " seconds." << endl;
            delete[] dist;
            return global_dist;
        }
    }
    
    // Get the original node ID from the internal index
    int getNodeId(int index) const {
        return nodeMap.getNodeId(index);
    }
    
    // Get number of vertices
    int getVertexCount() const { 
        return V; 
    }
};

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <num_processes> " << argv[0] << " <graph_file> [source_node=1]\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    string filename = argv[1];
    int sourceNodeId = (argc > 2) ? atoi(argv[2]) : 1;  // Default source node is 1
    
    // Create parallel graph
    ParallelGraph graph(rank, size);
    
    // Measure time for loading the graph
    double start_load = MPI_Wtime();
    if (!graph.loadFromFile(filename)) {
        MPI_Finalize();
        return 1;
    }
    double end_load = MPI_Wtime();
    
    if (rank == 0) {
        cout << "Graph loading time: " << end_load - start_load << " seconds\n";
        cout << "Running Parallel Bellman-Ford from source node " << sourceNodeId << "...\n";
    }
    
    // Barrier to synchronize all processes before starting the algorithm
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Measure time for Parallel Bellman-Ford algorithm
    double start_bf = MPI_Wtime();
    int* distances = graph.parallelBellmanFord(sourceNodeId);
    double end_bf = MPI_Wtime();
    
    // Only rank 0 has the final distances
    if (rank == 0) {
        if (distances == NULL) {
            cerr << "Error in Parallel Bellman-Ford algorithm.\n";
            MPI_Finalize();
            return 1;
        }
        
        double bf_time = end_bf - start_bf;
        
        cout << "\nPerformance Metrics:\n";
        cout << "-------------------\n";
        cout << "Number of processes: " << size << "\n";
        cout << "Total nodes processed: " << graph.getVertexCount() << "\n";
        cout << "Parallel Bellman-Ford execution time: " << bf_time << " seconds\n";
        cout << "Nodes processed per second: " << graph.getVertexCount() / bf_time << "\n";
        
        // Print the shortest distances to a few sample nodes
        cout << "\nSample of shortest distances from node " << sourceNodeId << ":\n";
        int sampleSize = min(20, graph.getVertexCount());
        for (int i = 0; i < sampleSize; i++) {
            int nodeId = graph.getNodeId(i);
            if (distances[i] == INT_MAX) {
                cout << "Node " << nodeId << ": INFINITY\n";
            } else {
                cout << "Node " << nodeId << ": " << distances[i] << "\n";
            }
        }
        
        // Output summary stats of distances
        int maxDist = 0;
        int reachableNodes = 0;
        long long sumDist = 0;
        
        for (int i = 0; i < graph.getVertexCount(); i++) {
            if (distances[i] != INT_MAX) {
                maxDist = max(maxDist, distances[i]);
                reachableNodes++;
                sumDist += distances[i];
            }
        }
        
        cout << "\nDistance Statistics:\n";
        cout << "-------------------\n";
        cout << "Maximum distance: " << maxDist << "\n";
        cout << "Reachable nodes: " << reachableNodes << " out of " << graph.getVertexCount() << "\n";
        if (reachableNodes > 0) {
            cout << "Average distance to reachable nodes: " << (double)sumDist / reachableNodes << "\n";
        }
        
        // Clean up
        delete[] distances;
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
