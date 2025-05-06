#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm> // Added for std::find
#include <mpi.h>

using namespace std;

// Constants for the graph
const int MAX_VERTICES = 4000000;   // Maximum number of vertices
const int MAX_EDGES = 250000000;    // Maximum number of edges

// Structure to represent an edge in the graph
struct Edge {
    int src, dest, weight;
};

// Structure to maintain node ID mapping
struct NodeMapping {
    int* nodeIdToIndex;
    int* indexToNodeId;
    int nextIndex;
    
    NodeMapping() {
        nodeIdToIndex = new int[MAX_VERTICES * 2];
        indexToNodeId = new int[MAX_VERTICES];
        nextIndex = 0;
        
        for (int i = 0; i < MAX_VERTICES * 2; i++) {
            nodeIdToIndex[i] = -1;
        }
    }
    
    ~NodeMapping() {
        delete[] nodeIdToIndex;
        delete[] indexToNodeId;
    }
    
    // Get index for a node ID, creating a new index if needed
    int getIndex(int nodeId) {
        int hash = nodeId % (MAX_VERTICES * 2);
        if (hash < 0) hash += (MAX_VERTICES * 2);
        
        while (true) {
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
    
    // Get the original node ID from an index
    int getNodeId(int index) const {
        if (index >= 0 && index < nextIndex) {
            return indexToNodeId[index];
        }
        return -1;
    }
    
    // Get the total number of vertices
    int getVertexCount() {
        return nextIndex;
    }
};

// Class to represent a graph partition
class GraphPartition {
private:
    vector<Edge> edges;               // Edges for this partition
    vector<Edge> ghostEdges;          // Edges connecting to other partitions
    vector<int> localVertices;        // Vertices in this partition
    vector<int> ghostVertices;        // Vertices in other partitions connected to this one
    NodeMapping nodeMap;
    int localVertexCount;             // Number of vertices in this partition
    int totalVertices;                // Total number of vertices across all partitions
    int rank;                         // MPI rank
    int size;                         // MPI size (number of processes)
    vector<int> vertexToPartition;    // Maps vertex index to partition ID

public:
    GraphPartition(int mpiRank, int mpiSize) 
        : rank(mpiRank), size(mpiSize), localVertexCount(0), totalVertices(0) {
        vertexToPartition.resize(MAX_VERTICES, -1);
        cout << "Process " << rank << " initialized GraphPartition" << endl;
    }
    
    // Load partition information from METIS output file
    bool loadPartitionInfo(const string& partitionFile) {
        cout << "Process " << rank << " starting to load partition info from " << partitionFile << endl;
        
        ifstream file(partitionFile.c_str());
        if (!file.is_open()) {
            if (rank == 0) {
                cerr << "Error: Unable to open partition file " << partitionFile << endl;
            }
            return false;
        }
        
        int nodeId = 0;
        int partId;
        int nodesProcessed = 0;
        int nodesInThisPartition = 0;
        
        while (file >> partId) {
            if (partId >= size) {
                if (rank == 0) {
                    cerr << "Error: Partition ID " << partId << " exceeds number of processes " << size << endl;
                }
                file.close();
                return false;
            }
            
            int nodeIndex = nodeMap.getIndex(nodeId);
            vertexToPartition[nodeIndex] = partId;
            
            if (partId == rank) {
                localVertices.push_back(nodeIndex);
                nodesInThisPartition++;
            }
            
            nodeId++;
            nodesProcessed++;
            
            // Print progress every million nodes
            if (rank == 0 && nodesProcessed % 100000 == 0) {
                cout << "Processed " << nodesProcessed << " nodes from partition file..." << endl;
            }
        }
        
        localVertexCount = localVertices.size();
        cout << "Process " << rank << " loaded partition info: " << nodesProcessed << " total nodes, " 
             << nodesInThisPartition << " nodes in this partition" << endl;
        
        // Gather and print distribution info
        int* localCounts = new int[size];
        for (int i = 0; i < size; i++) {
            localCounts[i] = 0;
        }
        localCounts[rank] = nodesInThisPartition;
        
        int* globalCounts = NULL;
        if (rank == 0) {
            globalCounts = new int[size];
        }
        
        MPI_Reduce(localCounts, globalCounts, size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0 && globalCounts) {
            cout << "Node distribution across partitions:" << endl;
            for (int i = 0; i < size; i++) {
                cout << "  Partition " << i << ": " << globalCounts[i] << " nodes" << endl;
            }
            delete[] globalCounts;
        }
        
        delete[] localCounts;
        file.close();
        return true;
    }
    
    // Load the graph data for this partition
    bool loadPartitionedGraph(const string& graphFile) {
        cout << "Process " << rank << " starting to load graph from " << graphFile << endl;
        
        ifstream file(graphFile.c_str());
        if (!file.is_open()) {
            if (rank == 0) {
                cerr << "Error: Unable to open graph file " << graphFile << endl;
            }
            return false;
        }
        
        string line;
        // Read first line containing graph information
        getline(file, line);
        istringstream headerStream(line);
        int nvtxs, nedges;
        
        // METIS graph format: first line contains nvtxs and nedges (edges counted once)
        if (!(headerStream >> nvtxs >> nedges)) {
            if (rank == 0) {
                cerr << "Error: Invalid graph header format" << endl;
            }
            file.close();
            return false;
        }
        
        if (rank == 0) {
            cout << "Reading METIS graph with " << nvtxs << " vertices and " << nedges << " edges" << endl;
        }
        
        // Read adjacency list for each vertex
        int nodesProcessed = 0;
        int localEdgesAdded = 0;
        int ghostEdgesAdded = 0;
        
        for (int i = 0; i < nvtxs; i++) {
            if (!getline(file, line)) {
                if (rank == 0) {
                    cerr << "Error: Unexpected end of file at vertex " << i << endl;
                }
                file.close();
                return false;
            }

            // Print loading progress
            if (rank == 0 && i % 50000 == 0) {
                cout << "Loading graph: processed " << i << " of " << nvtxs << " vertices (" 
                     << (100.0 * i / nvtxs) << "%)" << endl;
            }
            
            istringstream ss(line);
            int neighbor, weight;
            
            // Node IDs in METIS format are 1-based, but we use 0-based internally
            int nodeId = i;
            int nodeIndex = nodeMap.getIndex(nodeId);
            
            // Check if this vertex belongs to this partition
            int partId = vertexToPartition[nodeIndex];
            bool isLocalVertex = (partId == rank);
            
            if (isLocalVertex) {
                nodesProcessed++;
            }
            
            // Read adjacency list
            while (ss >> neighbor) {
                // Check for optional edge weight (we'll ignore it and use weight=1)
                if (ss >> weight) {
                    // Has weight - in weighted graph format
                } else {
                    // No weight - in unweighted graph format
                    weight = 1;
                }
                
                // Adjust from 1-based to 0-based indexing if needed
                neighbor--; // METIS uses 1-based indexing
                
                // Add edge to graph
                int neighborIndex = nodeMap.getIndex(neighbor);
                int neighborPartId = vertexToPartition[neighborIndex];
                
                Edge e;
                e.src = nodeIndex;
                e.dest = neighborIndex;
                e.weight = weight;
                
                // If edge is fully in this partition
                if (isLocalVertex && neighborPartId == rank) {
                    edges.push_back(e);
                    localEdgesAdded++;
                }
                // If edge crosses partition boundaries and involves this partition
                else if (isLocalVertex || neighborPartId == rank) {
                    ghostEdges.push_back(e);
                    ghostEdgesAdded++;
                    
                    // If the neighbor is in another partition, add it to ghost vertices
                    if (isLocalVertex && neighborPartId != rank) {
                        int ghostVertex = neighborIndex;
                        if (find(ghostVertices.begin(), ghostVertices.end(), ghostVertex) == ghostVertices.end()) {
                            ghostVertices.push_back(ghostVertex);
                        }
                    }
                }
            }
            
            // Print detailed progress for each process occasionally
            if (i % 200000 == 0 || i == nvtxs - 1) {
                cout << "Process " << rank << " progress: " << nodesProcessed << " local nodes processed, " 
                     << localEdgesAdded << " local edges, " << ghostEdgesAdded << " ghost edges, " 
                     << ghostVertices.size() << " ghost vertices" << endl;
            }
        }
        
        totalVertices = nodeMap.getVertexCount();
        
        cout << "Process " << rank << " graph loading complete:" << endl
             << "  - Total vertices in graph: " << totalVertices << endl
             << "  - Local vertices: " << localVertexCount << endl
             << "  - Local edges: " << edges.size() << endl
             << "  - Ghost edges (connecting to other partitions): " << ghostEdges.size() << endl
             << "  - Ghost vertices: " << ghostVertices.size() << endl;
        
        // Gather and report overall statistics
        if (rank == 0) {
            cout << "\nGraph loaded successfully across all processes." << endl;
            
            // Get total edge counts from all processes
            long long totalLocalEdges = 0;
            long long localEdgeCount = edges.size();
            MPI_Reduce(&localEdgeCount, &totalLocalEdges, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            
            long long totalGhostEdges = 0;
            long long ghostEdgeCount = ghostEdges.size();
            MPI_Reduce(&ghostEdgeCount, &totalGhostEdges, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            
            cout << "Total local edges across all processes: " << totalLocalEdges << endl;
            cout << "Total ghost edges across all processes: " << totalGhostEdges << endl;
            cout << "Average local edges per process: " << (double)totalLocalEdges / size << endl;
            cout << "Average ghost edges per process: " << (double)totalGhostEdges / size << endl;
        }
        
        file.close();
        return true;
    }
    
    // Process an edge, determining if it's local or connects to another partition
    void processEdge(int srcId, int destId) {
        int srcIndex = nodeMap.getIndex(srcId);
        int destIndex = nodeMap.getIndex(destId);
        
        // Determine which partition this edge belongs to
        int srcPartition = (vertexToPartition[srcIndex] != -1) ? vertexToPartition[srcIndex] : rank;
        int destPartition = (vertexToPartition[destIndex] != -1) ? vertexToPartition[destIndex] : rank;
        
        Edge e;
        e.src = srcIndex;
        e.dest = destIndex;
        e.weight = 1;  // Default weight
        
        // If edge is fully in this partition
        if (srcPartition == rank && destPartition == rank) {
            edges.push_back(e);
            
            // Add reverse edge for undirected graph
            Edge rev;
            rev.src = destIndex;
            rev.dest = srcIndex;
            rev.weight = 1;
            edges.push_back(rev);
        }
        // If edge crosses partition boundaries and involves this partition
        else if (srcPartition == rank || destPartition == rank) {
            ghostEdges.push_back(e);
            
            // Add reverse edge for undirected graph
            Edge rev;
            rev.src = destIndex;
            rev.dest = srcIndex;
            rev.weight = 1;
            ghostEdges.push_back(rev);
        }
    }
    
    // Parallel Bellman-Ford algorithm
    void parallelBellmanFord(int sourceNodeId) {
        cout << "Process " << rank << " starting Bellman-Ford with source node " << sourceNodeId << endl;
        
        int sourceIndex = -1;
        
        // Find the index for the source node ID
        for (int i = 0; i < totalVertices; i++) {
            if (nodeMap.getNodeId(i) == sourceNodeId) {
                sourceIndex = i;
                break;
            }
        }
        
        // If source not found, broadcast error
        int sourceFound = (sourceIndex != -1) ? 1 : 0;
        int globalSourceFound;
        MPI_Allreduce(&sourceFound, &globalSourceFound, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        if (globalSourceFound == 0) {
            if (rank == 0) {
                cerr << "Error: Source node " << sourceNodeId << " not found in the graph." << endl;
            }
            return;
        }
        
        // Share the source index with all processes
        MPI_Bcast(&sourceIndex, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Identify which process owns the source vertex
        int sourceOwner = vertexToPartition[sourceIndex];
        if (rank == 0) {
            cout << "Source node " << sourceNodeId << " (index " << sourceIndex << ") is owned by process " 
                 << sourceOwner << endl;
        }
        
        // Initialize distance array for all vertices
        vector<int> distances(totalVertices, INT_MAX);
        
        // The process owning the source node initializes its distance to 0
        int sourcePartition = vertexToPartition[sourceIndex];
        if (sourcePartition == rank) {
            distances[sourceIndex] = 0;
            cout << "Process " << rank << " initializing source distance to 0" << endl;
        }
        
        // Arrays for sending and receiving distance updates
        vector<int> sendDistances(totalVertices, INT_MAX);
        vector<int> recvDistances(totalVertices, INT_MAX);
        
        bool anyUpdate = true;
        int iteration = 0;
        int maxIterations = totalVertices - 1;  // Maximum number of iterations needed
        
        // Report maximum possible iterations
        if (rank == 0) {
            cout << "Maximum iterations needed: " << maxIterations << endl;
        }
        
        // Track relaxation stats
        long long totalRelaxations = 0;
        long long successfulRelaxations = 0;
        
        // Bellman-Ford main loop
        while (anyUpdate && iteration < maxIterations) {
            iteration++;
            anyUpdate = false;
            bool localUpdate = false;
            long long iterRelaxations = 0;
            long long iterSuccessful = 0;
            
            // Print detailed progress for each process at beginning of iteration
            cout << "Process " << rank << " starting iteration " << iteration << endl;
            
            // Step 1: Synchronize distances across all processes
            for (size_t i = 0; i < distances.size(); i++) {
                sendDistances[i] = distances[i];
            }
            
            MPI_Allreduce(&sendDistances[0], &recvDistances[0], totalVertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            
            // Update local distances with received global distances
            int numUpdatesFromSync = 0;
            for (size_t i = 0; i < distances.size(); i++) {
                if (recvDistances[i] < distances[i]) {
                    distances[i] = recvDistances[i];
                    localUpdate = true;
                    numUpdatesFromSync++;
                }
            }
            
            cout << "Process " << rank << " received " << numUpdatesFromSync << " distance updates from sync" << endl;
            
            // Step 2: Relax edges in local partition
            int localEdgeRelaxations = 0;
            for (size_t i = 0; i < edges.size(); i++) {
                int u = edges[i].src;
                int v = edges[i].dest;
                int weight = edges[i].weight;
                
                iterRelaxations++;
                
                if (distances[u] != INT_MAX && distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    localUpdate = true;
                    iterSuccessful++;
                    localEdgeRelaxations++;
                }
            }
            
            // Step 3: Relax ghost edges connecting to other partitions
            int ghostEdgeRelaxations = 0;
            for (size_t i = 0; i < ghostEdges.size(); i++) {
                int u = ghostEdges[i].src;
                int v = ghostEdges[i].dest;
                int weight = ghostEdges[i].weight;
                
                iterRelaxations++;
                
                if (distances[u] != INT_MAX && distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    localUpdate = true;
                    iterSuccessful++;
                    ghostEdgeRelaxations++;
                }
            }
            
            totalRelaxations += iterRelaxations;
            successfulRelaxations += iterSuccessful;
            
            cout << "Process " << rank << " completed iteration " << iteration << ": " 
                 << localEdgeRelaxations << " local edge relaxations, "
                 << ghostEdgeRelaxations << " ghost edge relaxations" << endl;
            
            // Check if any process made an update
            MPI_Allreduce(&localUpdate, &anyUpdate, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            
            // Count reachable vertices in this iteration
            int localReachable = 0;
            for (size_t i = 0; i < distances.size(); i++) {
                if (distances[i] != INT_MAX) {
                    localReachable++;
                }
            }
            
            int globalReachable;
            MPI_Reduce(&localReachable, &globalReachable, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
            
            // Print progress from rank 0
            if (rank == 0) {
                cout << "Completed iteration " << iteration << " of max " << maxIterations 
                     << " (" << (100.0 * iteration / maxIterations) << "%)" << endl;
                cout << "Nodes reachable so far: " << globalReachable << " of " << totalVertices 
                     << " (" << (100.0 * globalReachable / totalVertices) << "%)" << endl;
                
                if (!anyUpdate) {
                    cout << "No updates in this iteration - algorithm converged early" << endl;
                }
            }
            
            // Print more detailed progress every few iterations
            if (iteration % 10 == 0 || !anyUpdate || iteration == maxIterations) {
                cout << "Process " << rank << " after " << iteration << " iterations: " 
                     << "Total edge relaxations: " << totalRelaxations 
                     << ", Successful: " << successfulRelaxations << endl;
            }
        }
        
        cout << "Process " << rank << " finished Bellman-Ford after " << iteration << " iterations" << endl;
        
        if (rank == 0) {
            if (iteration < maxIterations) {
                cout << "Early convergence at iteration " << iteration << " of " << maxIterations << endl;
            }
            
            // Print sample of shortest distances
            cout << "\nSample of shortest distances from node " << sourceNodeId << ":" << endl;
            int sampleSize = min(20, totalVertices);
            for (int i = 0; i < sampleSize; i++) {
                int nodeId = nodeMap.getNodeId(i);
                if (distances[i] == INT_MAX) {
                    cout << "Node " << nodeId << ": INFINITY" << endl;
                } else {
                    cout << "Node " << nodeId << ": " << distances[i] << endl;
                }
            }
            
            // Output summary stats of distances
            int maxDist = 0;
            int reachableNodes = 0;
            long long sumDist = 0;
            
            for (int i = 0; i < totalVertices; i++) {
                if (distances[i] != INT_MAX) {
                    maxDist = max(maxDist, distances[i]);
                    reachableNodes++;
                    sumDist += distances[i];
                }
            }
            
            cout << "\nDistance Statistics:" << endl;
            cout << "-------------------" << endl;
            cout << "Maximum distance: " << maxDist << endl;
            cout << "Reachable nodes: " << reachableNodes << " out of " << totalVertices << endl;
            if (reachableNodes > 0) {
                cout << "Average distance to reachable nodes: " << (double)sumDist / reachableNodes << endl;
            }
        }
    }
    
    // Get the total number of vertices
    int getVertexCount() const {
        return totalVertices;
    }
};

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Print basic information about the MPI job
    cout << "Process " << rank << " of " << size << " started" << endl;
    
    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <processes> " << argv[0] << " <graph_file> <partition_file> [source_node=1]" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    string graphFile = argv[1];
    string partitionFile = argv[2];
    int sourceNodeId = (argc > 3) ? atoi(argv[3]) : 1;  // Default source node is 1
    
    if (rank == 0) {
        cout << "Graph file: " << graphFile << endl;
        cout << "Partition file: " << partitionFile << endl;
        cout << "Source node: " << sourceNodeId << endl;
        cout << "Number of processes: " << size << endl;
    }
    
    // Create graph partition for this process
    GraphPartition graph(rank, size);
    
    // Measure time for loading the graph
    double start_load = MPI_Wtime();
    
    // First load partition information
    if (!graph.loadPartitionInfo(partitionFile)) {
        MPI_Finalize();
        return 1;
    }
    
    cout << "Process " << rank << " successfully loaded partition information" << endl;
    
    // Then load the actual graph data for this partition
    if (!graph.loadPartitionedGraph(graphFile)) {
        MPI_Finalize();
        return 1;
    }
    
    double end_load = MPI_Wtime();
    double load_time = end_load - start_load;
    
    cout << "Process " << rank << " graph loading completed in " << load_time << " seconds" << endl;
    
    // Measure time for parallel Bellman-Ford algorithm
    if (rank == 0) {
        cout << "Graph loading time: " << load_time << " seconds" << endl;
        cout << "Running parallel Bellman-Ford from source node " << sourceNodeId << "..." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before starting the algorithm
    cout << "Process " << rank << " passed barrier, starting Bellman-Ford" << endl;
    
    double start_bf = MPI_Wtime();
    graph.parallelBellmanFord(sourceNodeId);
    double end_bf = MPI_Wtime();
    double bf_time = end_bf - start_bf;
    
    cout << "Process " << rank << " completed Bellman-Ford in " << bf_time << " seconds" << endl;
    
    if (rank == 0) {
        cout << "\nPerformance Metrics:" << endl;
        cout << "-------------------" << endl;
        cout << "Total nodes processed: " << graph.getVertexCount() << endl;
        cout << "Parallel Bellman-Ford execution time: " << bf_time << " seconds" << endl;
        cout << "Nodes processed per second: " << graph.getVertexCount() / bf_time << endl;
        cout << "Number of processes: " << size << endl;
    }
    
    MPI_Finalize();
    cout << "Process " << rank << " finalized MPI and exiting" << endl;
    return 0;
}
