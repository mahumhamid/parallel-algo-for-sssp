#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <cmath>

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

class Graph {
private:
    Edge* edges;
    int E;    // Current number of edges
    int V;    // Current number of vertices
    NodeMapping nodeMap;

public:
    Graph() {
        edges = new Edge[MAX_EDGES];
        E = 0;
        V = 0;
    }
    
    ~Graph() {
        delete[] edges;
    }
    
    void addEdge(int src, int dest, int weight = 1) {
        int srcIndex = nodeMap.getIndex(src);
        int destIndex = nodeMap.getIndex(dest);
        
        V = nodeMap.getVertexCount();
        
        edges[E].src = srcIndex;  //add edge
        edges[E].dest = destIndex;
        edges[E].weight = weight;
        E++;
        
        edges[E].src = destIndex;  //add reverse edge
        edges[E].dest = srcIndex;
        edges[E].weight = weight;
        E++;
    }
    
    bool loadFromFile(const string& filename) {
        ifstream file(filename.c_str());
        if (!file.is_open()) {
            cerr << "Error: Unable to open file " << filename << endl;
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
            
            // Print progress every million edges
            if (lineCount % 1000000 == 0) {
                cout << "Processed " << lineCount << " edges...\n";
            }
        }
        
        cout << "\nGraph loaded successfully.\n";
        cout << "Number of vertices: " << V << "\n";
        cout << "Number of undirected edges: " << E/2 << "\n";
        
        file.close();
        return true;
    }
    
    // Bellman-Ford algorithm for SSSP
    int* bellmanFord(int sourceNodeId) {
        int sourceIndex = -1;
        
        // Find the index for the source node ID
        for (int i = 0; i < V; i++) {
            if (nodeMap.getNodeId(i) == sourceNodeId) {
                sourceIndex = i;
                break;
            }
        }
        
        if (sourceIndex == -1) {
            cerr << "Error: Source node " << sourceNodeId << " not found in the graph.\n";
            return NULL;
        }
        
        // Initialize distance array
        int* dist = new int[V];
        for (int i = 0; i < V; i++) {
            dist[i] = INT_MAX;
        }
        dist[sourceIndex] = 0;
        
        // Relax all edges 
        for (int i = 1; i < V; i++) {
            bool anyUpdate = false;
            for (int j = 0; j < E; j++) {
                int u = edges[j].src;
                int v = edges[j].dest;
                int weight = edges[j].weight;
                
                if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    anyUpdate = true;
                }
            }
            
            if (!anyUpdate) {
                cout << "Early convergence at iteration " << i << " of " << (V - 1) << endl;
                break;
            }
            
            if (i % max(1, (V - 1) / 10) == 0) {
                cout << "Completed " << i << " of " << (V - 1) << " iterations (" 
                     << (100.0 * i / (V - 1)) << "%)\n";
            }
        }
        
        // Check for negative weight cycles
        for (int j = 0; j < E; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int weight = edges[j].weight;
            
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                cerr << "Graph contains negative weight cycle.\n";
                delete[] dist;
                return NULL;
            }
        }
        
        return dist;
    }
    
    // Get the original node ID from the internal index
    int getNodeId(int index) const {
        return nodeMap.getNodeId(index);
    }

    int getVertexCount() const { 
        return V; 
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [source_node=1]\n";
        return 1;
    }
    
    string filename = argv[1];
    int sourceNodeId = (argc > 2) ? atoi(argv[2]) : 1; 
    
    Graph graph;
    
    // Measure time for loading the graph
    clock_t start_load = clock();
    if (!graph.loadFromFile(filename)) {
        return 1;
    }
    clock_t end_load = clock();
    double load_time = ((double)(end_load - start_load)) / CLOCKS_PER_SEC;
    
    cout << "Graph loading time: " << load_time << " seconds\n";
    
    // Measure time for SSSP algorithm
    cout << "Running Bellman-Ford from source node " << sourceNodeId << "...\n";
    clock_t start_bf = clock();
    int* distances = graph.bellmanFord(sourceNodeId);
    clock_t end_bf = clock();
    double bf_time = ((double)(end_bf - start_bf)) / CLOCKS_PER_SEC;
    
    if (distances == NULL) {
        cerr << "Error in Bellman-Ford algorithm.\n";
        return 1;
    }
    
    cout << "\nPerformance Metrics:\n";
    cout << "-------------------\n";
    cout << "Total nodes processed: " << graph.getVertexCount() << "\n";
    cout << "Bellman-Ford execution time: " << bf_time << " seconds\n";
    cout << "Nodes processed per second: " << graph.getVertexCount() / bf_time << "\n";
    
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
    
    return 0;
}
