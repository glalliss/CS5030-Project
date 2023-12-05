# CS5030-Project

## 1. Serial
- Code with instructions on how to build and execute all the implementations
    - $ g++ serial.cpp -o serial
    - $ ./serial
- Description the approach used for each of the following implementations
    - Dataset from https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs
    - Adapted from https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/ converted to work with 3D

## 2. Parallel shared memory CPU
- Code with instructions on how to build and execute all the implementations
    - $ g++ -fopenmp omp_cpu.cpp -o omp
    - $ ./omp (number of threads)
- Description the approach used for each of the following implementations
    - 

## 3. Parallel shared memory GPU
- Code with instructions on how to build and execute all the implementations
    - 
- Description the approach used for each of the following implementations
    - 

## 4. Distributed memory CPU
- Code with instructions on how to build and execute all the implementations
    - $ mpic++ mpi_cpu.cpp -o mpi
    - $ mpiexec -n (number of threads) ./mpi
- Description the approach used for each of the following implementations
    - 

## 5. Distributed memory GPU
- Code with instructions on how to build and execute all the implementations
    - 
- Description the approach used for each of the following implementations
    - 

## Scaling study experiments where you compare implementations:
- 1 vs 2 vs 3 (note: you don't need a scaling study for GPUs, you can look instead at different block/tile size)
- 4 vs 5

## Validation Function
- Check that the result from parallel implementations is equal to the serial output implementation

## Visualization
- testing.py
