#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <limits.h>

#define MAX_VERTICES 100000
#define MAX_EDGES 500000

using namespace std;

// Helper function to check OpenCL errors
void check_cl_error(cl_int err, const char* message) {
    if (err != CL_SUCCESS) {
        cerr << message << " Error code: " << err << endl;
        exit(1);
    }
}

// Structure to represent an edge
struct Edge {
    int src, dest, weight;
};

// Function to load graph from a file
void load_graph(const string& filename, vector<Edge>& edges, int& numVertices) {
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream ss(line);
        int src, dest, weight;
        ss >> src >> dest >> weight;
        edges.push_back({src, dest, weight});
        numVertices = max(numVertices, max(src, dest) + 1);
    }
}

// OpenCL Setup
cl_program build_program(cl_context context, cl_device_id device, const char* filename) {
    ifstream file(filename);
    stringstream buffer;
    buffer << file.rdbuf();
    string source = buffer.str();
    const char* sourceCode = source.c_str();

    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &sourceCode, NULL, &err);
    check_cl_error(err, "Failed to create program");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    check_cl_error(err, "Failed to build program");

    return program;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./sssp_opencl <graph_file>" << endl;
        return -1;
    }

    string filename = argv[1];
    vector<Edge> edges;
    int numVertices = 0;
    load_graph(filename, edges, numVertices);
    int numEdges = edges.size();

    // Set up OpenCL environment
    cl_int err;
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    check_cl_error(err, "Failed to get platform ID");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_cl_error(err, "Failed to get device ID");

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_cl_error(err, "Failed to create context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    check_cl_error(err, "Failed to create command queue");

    // Create buffers
    cl_mem edgeStartBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, numEdges * sizeof(int), NULL, &err);
    check_cl_error(err, "Failed to create buffer for edgeStart");
    cl_mem edgeEndBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, numEdges * sizeof(int), NULL, &err);
    check_cl_error(err, "Failed to create buffer for edgeEnd");
    cl_mem edgeWeightsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, numEdges * sizeof(int), NULL, &err);
    check_cl_error(err, "Failed to create buffer for edgeWeights");
    cl_mem distancesBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, numVertices * sizeof(float), NULL, &err);
    check_cl_error(err, "Failed to create buffer for distances");
    cl_mem updatedBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    check_cl_error(err, "Failed to create buffer for updated");

    // Set initial distances to infinity, except for the source node
    vector<float> distances(numVertices, FLT_MAX);
    distances[0] = 0.0f; // Assuming source node is 0
    vector<int> updated(1, 0);

    // Write buffers to device
    err = clEnqueueWriteBuffer(queue, edgeStartBuffer, CL_TRUE, 0, numEdges * sizeof(int), &edges[0].src, 0, NULL, NULL);
    check_cl_error(err, "Failed to write edgeStart buffer");
    err = clEnqueueWriteBuffer(queue, edgeEndBuffer, CL_TRUE, 0, numEdges * sizeof(int), &edges[0].dest, 0, NULL, NULL);
    check_cl_error(err, "Failed to write edgeEnd buffer");
    err = clEnqueueWriteBuffer(queue, edgeWeightsBuffer, CL_TRUE, 0, numEdges * sizeof(int), &edges[0].weight, 0, NULL, NULL);
    check_cl_error(err, "Failed to write edgeWeights buffer");
    err = clEnqueueWriteBuffer(queue, distancesBuffer, CL_TRUE, 0, numVertices * sizeof(float), &distances[0], 0, NULL, NULL);
    check_cl_error(err, "Failed to write distances buffer");
    err = clEnqueueWriteBuffer(queue, updatedBuffer, CL_TRUE, 0, sizeof(int), &updated[0], 0, NULL, NULL);
    check_cl_error(err, "Failed to write updated buffer");

    // Build program and create kernel
    cl_program program = build_program(context, device, "bellman_ford.cl");
    cl_kernel kernel = clCreateKernel(program, "bellman_ford", &err);
    check_cl_error(err, "Failed to create kernel");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &edgeStartBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &edgeEndBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &edgeWeightsBuffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &distancesBuffer);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &updatedBuffer);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &numEdges);
    check_cl_error(err, "Failed to set kernel arguments");

    size_t global_work_size = numEdges;
    size_t local_work_size = 64; // Adjust as per your hardware

    // Run Bellman-Ford for a fixed number of iterations
    int maxIterations = numVertices - 1;
    for (int iter = 0; iter < maxIterations; ++iter) {
        updated[0] = 0;
        err = clEnqueueWriteBuffer(queue, updatedBuffer, CL_TRUE, 0, sizeof(int), &updated[0], 0, NULL, NULL);
        check_cl_error(err, "Failed to write updated buffer");

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        check_cl_error(err, "Failed to enqueue kernel");

        clFinish(queue);

        // Read the updated distances
        err = clEnqueueReadBuffer(queue, distancesBuffer, CL_TRUE, 0, numVertices * sizeof(float), &distances[0], 0, NULL, NULL);
        check_cl_error(err, "Failed to read distances buffer");

        // If no distances were updated, exit early
        err = clEnqueueReadBuffer(queue, updatedBuffer, CL_TRUE, 0, sizeof(int), &updated[0], 0, NULL, NULL);
        check_cl_error(err, "Failed to read updated buffer");

        if (updated[0] == 0) {
            break;
        }
    }

    // Print sample distances from source node
    if (distances[0] == FLT_MAX) {
        cout << "No path to source node." << endl;
    } else {
        cout << "Shortest path from node 0:" << endl;
        for (int i = 0; i < min(10, numVertices); ++i) {
            cout << "Node " << i << ": ";
            if (distances[i] == FLT_MAX) cout << "INFINITY\n";
            else cout << distances[i] << "\n";
        }
    }

    // Clean up
    clReleaseMemObject(edgeStartBuffer);
    clReleaseMemObject(edgeEndBuffer);
    clReleaseMemObject(edgeWeightsBuffer);
    clReleaseMemObject(distancesBuffer);
    clReleaseMemObject(updatedBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

