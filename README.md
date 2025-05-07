# Parallel Algorithms for Single Source Shortest Path (SSSP)

This project implements multiple versions of the Single Source Shortest Path (SSSP) algorithm using various parallel programming paradigms:
- Serial C++ Version  
- MPI Parallel Version  
- MPI + METIS Partitioned Version  
- MPI + OpenMP Version  

We evaluate the performance on the Orkut Social Network Graph dataset from the SNAP collection.

---

## Documentation

Full Project Report & Analysis:  
[Project Report](https://docs.google.com/document/d/1S3q8zTXWKhJrO4E3TpY2yd9NyFy_CdeH7BrN8MWFyuA/edit?usp=sharing)

---

## Directory Structure

Ensure all required files are placed in the same directory:

```
project-root/
├── serialcode.cpp
├── parallelmpi.cpp
├── parllelver.cpp
├── parallel_omp_mpi.cpp
├── preprocessing.py
├── com-orkut.ungraph.txt
├── orkut.graph
├── orkut.graph.part.4
```
---

## How to Run

### Serial Code

```bash
g++ -O2 -o test serialcode.cpp
./test com-orkut.ungraph.txt 1
```

---

### METIS Preprocessing

1. Install METIS:
```bash
sudo apt install metis
```

2. Run the preprocessing script:
```bash
python3 preprocessing.py
```

3. Partition the graph:
```bash
gpmetis orkut.graph 4
```

---

### MPI Version (parllelver.cpp)

```bash
mpic++ -Wall -O3 parllelver.cpp -o test -lm
mpirun -np 4 ./test com-orkut.ungraph.txt 1
```

---

### MPI + METIS Version (parallelmpi.cpp)

```bash
mpic++ -O3 -std=c++11 parallelmpi.cpp -o test
mpirun -np 4 ./test orkut.graph orkut.graph.part.4 1
```

---

### MPI + OpenMP Version (parallel_omp_mpi.cpp)

```bash
mpic++ -fopenmp -Wall -O3 parallel_omp_mpi.cpp -o test -lm
mpirun -np 4 ./test com-orkut.ungraph.txt 1
```

---

## Notes

- Place `com-orkut.ungraph.txt` in the same directory as your code.
- METIS preprocessing is only required for the MPI+METIS and MPI+OpenMP versions.
- Ensure OpenMP is supported in your compiler and OpenMPI is installed for MPI runs.

---

## Authors

Maryum Fasih  
**Team:** Maryum Fasih (22i0756), Mahum Hamid (22i1009), Abeer Jawad (22i1041)  
**Institute:** FAST-NUCES – Parallel & Distributed Computing Project
