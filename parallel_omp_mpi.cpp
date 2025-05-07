#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <omp.h>
#include <mpi.h>

using namespace std;

class Graph;

struct Edge {
    int src, dest, weight;
};

class NodeMapper {
private:
    unordered_map<int, int> nodeToIdx;
    vector<int> idxToNode;

public:
    int getIndex(int nodeId) {
        unordered_map<int, int>::iterator it = nodeToIdx.find(nodeId);
        if (it != nodeToIdx.end()) {
            return it->second;
        }
        
        int idx = idxToNode.size();
        nodeToIdx[nodeId] = idx;
        idxToNode.push_back(nodeId);
        return idx;
    }
    
    int getNodeId(int idx) const {
        if (idx >= 0 && idx < (int)idxToNode.size()) {
            return idxToNode[idx];
        }
        return -1;
    }
    
    int size() const {
        return idxToNode.size();
    }
};

class Graph {
private:
    vector<vector<pair<int, int>>> adjList;
    NodeMapper nodeMapper;
    int rank, numProcs;

public:
    Graph(MPI_Comm comm = MPI_COMM_WORLD) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &numProcs);
    }
    
    bool loadFromFile(const string& filename) {
        if (rank == 0) {
            cout << "Loading graph from file: " << filename << endl;
        }
        
        ifstream file(filename);
        if (!file.is_open()) {
            if (rank == 0) {
                cerr << "Error: Cannot open file " << filename << endl;
            }
            return false;
        }
        
        string line;
        int lineCount = 0;
        
        while (getline(file, line)) {
            if (line.empty() || line[0] != '#') {
                break;
            }
        }
        
        int src, dest, weight;
        weight = 1;
        
        if (!line.empty() && line[0] != '#') {
            istringstream iss(line);
            if (iss >> src >> dest) {
                if (!iss.eof()) {
                    iss >> weight;
                }
                
                if (lineCount % numProcs == rank) {
                    addEdge(src, dest, weight);
                }
                lineCount++;
            }
        }
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            istringstream iss(line);
            if (iss >> src >> dest) {
                weight = 1;
                if (!iss.eof()) {
                    iss >> weight;
                }
                
                if (lineCount % numProcs == rank) {
                    addEdge(src, dest, weight);
                }
                lineCount++;
            }
            
            if (rank == 0 && lineCount % 1000000 == 0) {
                cout << "Processed " << lineCount << " edges..." << endl;
            }
        }
        
        file.close();
        
        synchronizeGraph();
        
        if (rank == 0) {
            cout << "Graph loading completed." << endl;
            cout << "Number of vertices: " << adjList.size() << endl;
            
            int localEdges = 0;
            for (size_t i = 0; i < adjList.size(); ++i) {
                localEdges += adjList[i].size();
            }
            int totalEdges = 0;
            MPI_Reduce(&localEdges, &totalEdges, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            cout << "Number of directed edges: " << totalEdges << endl;
            cout << "Number of undirected edges: " << totalEdges / 2 << endl;
        } else {
            int localEdges = 0;
            for (size_t i = 0; i < adjList.size(); ++i) {
                localEdges += adjList[i].size();
            }
            MPI_Reduce(&localEdges, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        
        return true;
    }
    
    void addEdge(int src, int dest, int weight = 1) {
        int srcIdx = nodeMapper.getIndex(src);
        int destIdx = nodeMapper.getIndex(dest);
        
        #pragma omp critical
        {
            if (srcIdx >= (int)adjList.size()) {
                adjList.resize(srcIdx + 1);
            }
            if (destIdx >= (int)adjList.size()) {
                adjList.resize(destIdx + 1);
            }
            
            adjList[srcIdx].push_back(make_pair(destIdx, weight));
            adjList[destIdx].push_back(make_pair(srcIdx, weight));
        }
    }
    
    void synchronizeGraph() {
        int localVertices = nodeMapper.size();
        int maxVertices = 0;
        
        MPI_Allreduce(&localVertices, &maxVertices, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        if ((int)adjList.size() < maxVertices) {
            adjList.resize(maxVertices);
        }
    }
    
    vector<int> bellmanFord(int sourceNodeId) {
        int sourceIdx = -1;
        
        for (int i = 0; i < nodeMapper.size(); i++) {
            if (nodeMapper.getNodeId(i) == sourceNodeId) {
                sourceIdx = i;
                break;
            }
        }
        
        MPI_Allreduce(MPI_IN_PLACE, &sourceIdx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        if (sourceIdx == -1) {
            if (rank == 0) {
                cerr << "Error: Source node " << sourceNodeId << " not found in the graph." << endl;
            }
            return vector<int>();
        }
        
        int n = adjList.size();
        vector<int> dist(n, numeric_limits<int>::max());
        dist[sourceIdx] = 0;
        
        if (rank == 0) {
            cout << "Running Parallel Bellman-Ford from source node " << sourceNodeId << "..." << endl;
        }
        
        int chunkSize = (n + numProcs - 1) / numProcs;
        int startVertex = rank * chunkSize;
        int endVertex = min(startVertex + chunkSize, n);
        
        bool changed = true;
        int iterations = 0;
        int maxIterations = min(n, 100);
        
        while (changed && iterations < maxIterations) {
            iterations++;
            bool localChanged = false;
            
            MPI_Bcast(dist.data(), dist.size(), MPI_INT, 0, MPI_COMM_WORLD);
            
            #pragma omp parallel for reduction(||:localChanged)
            for (int u = startVertex; u < endVertex; u++) {
                if (dist[u] == numeric_limits<int>::max()) {
                    continue;
                }
                
                for (size_t i = 0; i < adjList[u].size(); ++i) {
                    int v = adjList[u][i].first;
                    int weight = adjList[u][i].second;
                    
                    if (dist[u] != numeric_limits<int>::max() && 
                        (dist[v] == numeric_limits<int>::max() || dist[u] + weight < dist[v])) {
                        #pragma omp critical
                        {
                            if (dist[u] + weight < dist[v]) {
                                dist[v] = dist[u] + weight;
                                localChanged = true;
                            }
                        }
                    }
                }
            }
            
            vector<int> tempDist(n);
            MPI_Allreduce(dist.data(), tempDist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            dist.swap(tempDist);
            
            MPI_Allreduce(&localChanged, &changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            
            if (rank == 0 && iterations % 10 == 0) {
                cout << "Completed " << iterations << " iterations" << endl;
            }
        }
        
        if (rank == 0) {
            if (!changed) {
                cout << "Early convergence at iteration " << iterations << endl;
            } else {
                cout << "Reached maximum iterations: " << iterations << endl;
            }
        }
        
        return dist;
    }
    
    int getNodeId(int idx) const {
        return nodeMapper.getNodeId(idx);
    }
    
    int getVertexCount() const {
        return adjList.size();
    }
    
    int getEdgeCount() const {
        int count = 0;
        for (size_t i = 0; i < adjList.size(); ++i) {
            count += adjList[i].size();
        }
        return count;
    }
};

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int num_threads = 8;
    char* omp_env = getenv("OMP_NUM_THREADS");
    if (omp_env) {
        num_threads = atoi(omp_env);
    }
    omp_set_num_threads(num_threads);
    
    if (argc < 2) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <graph_file> [source_node=1]" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    string filename = argv[1];
    int sourceNodeId = (argc > 2) ? atoi(argv[2]) : 1;
    
    Graph graph;
    
    double loadStart = MPI_Wtime();
    if (!graph.loadFromFile(filename)) {
        MPI_Finalize();
        return 1;
    }
    double loadEnd = MPI_Wtime();
    
    double bfStart = MPI_Wtime();
    vector<int> distances = graph.bellmanFord(sourceNodeId);
    double bfEnd = MPI_Wtime();
    
    if (distances.empty()) {
        if (rank == 0) {
            cerr << "Error in Bellman-Ford algorithm." << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        int localEdges = graph.getEdgeCount();
        int totalEdges = 0;
        MPI_Reduce(MPI_IN_PLACE, &localEdges, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        totalEdges = localEdges;
        
        cout << "\nPerformance Metrics:" << endl;
        cout << "-------------------" << endl;
        cout << "Total nodes processed: " << graph.getVertexCount() << endl;
        cout << "Total edges processed: " << totalEdges << endl;
        cout << "Number of MPI processes: " << size << endl;
        cout << "Number of OpenMP threads: " << omp_get_max_threads() << endl;
        cout << "Graph loading time: " << (loadEnd - loadStart) << " seconds" << endl;
        cout << "Bellman-Ford execution time: " << (bfEnd - bfStart) << " seconds" << endl;
        cout << "Nodes processed per second: " << graph.getVertexCount() / (bfEnd - bfStart) << endl;
        cout << "Edges processed per second: " << totalEdges / (bfEnd - bfStart) << endl;
        
        cout << "\nSample of shortest distances from node " << sourceNodeId << ":" << endl;
        int sampleSize = min(20, (int)graph.getVertexCount());
        for (int i = 0; i < sampleSize; i++) {
            int nodeId = graph.getNodeId(i);
            if (distances[i] == numeric_limits<int>::max()) {
                cout << "Node " << nodeId << ": INFINITY" << endl;
            } else {
                cout << "Node " << nodeId << ": " << distances[i] << endl;
            }
        }
        
        int maxDist = 0;
        int reachableNodes = 0;
        int sumDist = 0;
        
        #pragma omp parallel for reduction(max:maxDist) reduction(+:reachableNodes,sumDist)
        for (int i = 0; i < (int)distances.size(); i++) {
            if (distances[i] != numeric_limits<int>::max()) {
                maxDist = max(maxDist, distances[i]);
                reachableNodes++;
                sumDist += distances[i];
            }
        }
        
        cout << "\nDistance Statistics:" << endl;
        cout << "-------------------" << endl;
        cout << "Maximum distance: " << maxDist << endl;
        cout << "Reachable nodes: " << reachableNodes << " out of " << graph.getVertexCount() << endl;
        if (reachableNodes > 0) {
            cout << "Average distance to reachable nodes: " << (double)sumDist / reachableNodes << endl;
        }
    } else {
        int localEdges = graph.getEdgeCount();
        MPI_Reduce(&localEdges, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}
