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
    - The data is read in by the main thread, then open mp is initiated for the k means algorithm. Unfortunately, the best stratagy we could come up with was to create threads for each epoch of the k means clustering, which most likely has some overhead. Once the algorithm is finished, the threads are killed and the main thread writes the output data.

## 3. Parallel shared memory GPU
- Code with instructions on how to build and execute all the implementations
    - nvcc kmeans.cu -o kmeans
    - ./kmeans
- Description the approach used for each of the following implementations
    - Using cuda we created three kernels to do the kmeans clustering. assignPointsToClusters(kernel function to assign clusters to points), computeNewCentroids(helper kernel function to compute the mean for each cluster) and updateNewCentroids(kernel function to compute the mean for each cluster and update its centroid). kmeans.cu utilizes global gpu memory. The kmeans algorithm is run for 100 epochs and for k = 5.
## 4. Distributed memory CPU
- Code with instructions on how to build and execute all the implementations
    - $ mpic++ mpi_cpu.cpp -o mpi
    - $ mpiexec -n (number of threads) ./mpi
- Description the approach used for each of the following implementations
    - With MPI, we needed to get data to each process. We ran out of time to do parallel file reading, so process 0 reads the file and distributes it to the other processes. Then, each process performs the k means algorithm on it's part of the data. Each epoch, data must be sent back to process 0 to compute the new centroids, then the centroids are brodcasted to all processes. Process 0 writes the completed data to the output file, and the program is complete.

## 5. Distributed memory GPU
- Code with instructions on how to build and execute all the implementations
    - module load cuda/12.
    - module load gcc/8.5.0
    - module load openmpi
    - nvcc -I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib -lmpi kmeans_mpi_gpu.cu -o kmeans_mpi
      use the prefix /uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-nehalem/gcc-8.5.0/openmpi-4.1.4-4a4yd73rjd4bjfpndftt2z22ljffgy56/ instead of /usr/mpi/gcc/openmpi-1.4.6/. You can use the command ompi_info to get your prefix.
- Description the approach used for each of the following implementations
    - 

## Scaling study experiments where you compare implementations:
All of these were ran on 100 epochs

# Serial: 
32.126825 seconds

# OpenMP:

| Number of cores | Time for mpi section |
| --- | --- |
| 2 | 56.696950 |
| 4 | 81.555408 |
| 8 | 92.022627 |
| 16 | 70.868146 | 
| 32 | 45.040339 |

There is probably an issue with our implementation that is making it take this long. Based on this trend, it is possible that an even higher number of cores would begin to show impovement over the serial version.

MPI: 

| Number of cores | Time for mpi section |
| --- | --- |
| 2 | 25.305014 |
| 4 | 18.232265 |
| 8 | 14.718279 |
| 16 | 13.123988 | 
| 32 | 12.392990 |

MPI had a large improvement over the serial version. We can see that as the number of cores increased, the speedup obtained decreased.

# GPU:

- 1 vs 2 vs 3 (note: you don't need a scaling study for GPUs, you can look instead at different block/tile size)
  Timimg results for shared memory GPU
    | Block Size | Time(secs) |
    | --- | --- |
    | 16 | 57.5286 |
    | 32 | 57.5273 |
    | 64 | 57.5267 |
    | 128 | 57.5132 | 
    | 256 | 57.5223 |
- 4 vs 5
  Timimg results for shared memory GPU
    | Number of Process | Time(secs) |
    | --- | --- |
    | 1 | 58.1139 |
    | 2 | 29.9078 |
    | 3 | 57.5267 |
    | 4 | 57.5132 |

## Validation Function
- Check that the result from parallel implementations is equal to the serial output implementation. Only run this after producing output from all the other programs.
    - $ g++ validate.cpp -o validate
    - $ ./validate

## Visualization
- visualization of the output
    - $ python visualize.py
    - View output png
    - Run AFTER one of the kmeans implementations.
