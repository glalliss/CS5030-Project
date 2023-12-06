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
    - The data is read in by the main thread, then open mpi is initiated for the k means algorithm. Unfortunately, the best stratagy we could come up with was to create threads for each epoch of the k means clustering, which most likely has some overhead. Once the algorithm is finished, the threads are killed and the main thread writes the output data.

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
    - With MPI, we needed to get data to each process. We ran out of time to do parallel file reading, so process 0 reads the file and distributes it to the other processes. Then, each process performs the k means algorithm on it's part of the data. Each epoch, data must be sent back to process 0 to compute the new centroids, then the centroids are brodcasted to all processes. Process 0 writes the completed data to the output file, and the program is complete.

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
    - $ g++ validate.cpp -o validate
    - $ ./validate

## Visualization
- visualization of the output
    - $ python visualize.py
    - View output png
