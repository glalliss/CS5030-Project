#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

const int k = 5;
const int epochs = 100;
const int TILE_WIDTH = 256;
struct Point {
    double x, y, z; // coordinates
    int cluster;    // no default cluster
    double minDist; // default infinite distance to nearest cluster
    // Initialize a point
    __host__ __device__ Point() :
        x(0.0), y(0.0), z(0.0), cluster(-1), minDist(std::numeric_limits<double>::max()) {}
    __host__ __device__ Point(double x, double y, double z) :
        x(x), y(y), z(z), cluster(-1), minDist(std::numeric_limits<double>::max()) {}
    
};
// Computes the (square) euclidean distance between this point and another
__device__ double distance(Point p1, Point p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) +
           (p1.y - p2.y) * (p1.y - p2.y) +
           (p1.z - p2.z) * (p1.z - p2.z);
}

__global__ void assignPointsToClusterstiled(Point* points, int numPoints, Point* centroids, int k, double maxout) {
    __shared__ Point tile[TILE_WIDTH];
    //__shared__ Point tile1[k];
    int tx = threadIdx.x;
    int index = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (index < numPoints) {
        double minDist = maxout;
        int clusterId = -1;
        //for (int i = 0; i < k; ++i) {
            //tile1[i] = centroids[i];
        //}
        tile[tx] = points[index];
        __syncthreads();

        for (int i = 0; i < k; ++i) {
            double dist = distance(tile[tx], centroids[i]);
            if (dist < minDist) {
                minDist = dist;
                clusterId = i;
            }
        }
        __syncthreads();

        points[index].minDist = minDist;
        points[index].cluster = clusterId;
    }
}

__global__ void computeNewCentroids(Point* points, int numPoints, Point* centroids, int k, int* nPoints, double* sumX, double* sumY, double* sumZ) {
    int clusterId = blockIdx.x * blockDim.x + threadIdx.x;
    if (clusterId < k) {
        for (int index = 0; index < numPoints ; index++)
        {
            if(points[index].cluster == clusterId){
                nPoints[clusterId] += 1;
                sumX[clusterId] += points[index].x;
                sumY[clusterId] += points[index].y;
                sumZ[clusterId] += points[index].z;

            }
        }
        centroids[clusterId].x = sumX[clusterId] / nPoints[clusterId];
        centroids[clusterId].y = sumY[clusterId] / nPoints[clusterId];
        centroids[clusterId].z = sumZ[clusterId] / nPoints[clusterId];
    }
}

__global__ void updateNewCentroids(Point* centroids,int* nPoints, double* sumX, double* sumY, double* sumZ) {
    int clusterId = blockIdx.x * blockDim.x + threadIdx.x;
    centroids[clusterId].x = sumX[clusterId] / nPoints[clusterId];
    centroids[clusterId].y = sumY[clusterId] / nPoints[clusterId];
    centroids[clusterId].z = sumZ[clusterId] / nPoints[clusterId];
}


// Reads in the data.csv file into a vector of points and return vector of points
std::vector<Point> readcsv()
{
    std::vector<Point> points;
    std::ifstream file("tracks_features.csv");
    std::string line;
    int danceabilityIndex = 9;
    int energyIndex = 10;
    int valenceIndex = 18;
    while (getline(file, line))
    {
        std::stringstream lineStream(line);
        std::vector<std::string> columns;
        while (!lineStream.eof())
        {
            std::string column;
            getline(lineStream, column, ',');
            columns.push_back(column);
            // Handle cases where a column contains a comma inside double quotes or a list
            if (columns.back().front() == '"' && std::count(columns.back().begin(), columns.back().end(), '"') % 2 != 0)
            {
                while (columns.back().front() == '"' && std::count(columns.back().begin(), columns.back().end(), '"') % 2 != 0)
                {
                    std::string nextColumn;
                    getline(lineStream, nextColumn, ',');
                    columns.back() += "," + nextColumn;
                    // std::cout << columns.back() << std::endl;
                }
            }
        }
        try
        {
            // std::cout << columns[danceabilityIndex] << "    " << columns[energyIndex] << "    " << columns[valenceIndex] << std::endl;
            double x, y, z;
            // Convert specific columns into a double to create a Point and add it to the points vector
            x = stod(columns[danceabilityIndex]);
            y = stod(columns[energyIndex]);
            z = stod(columns[valenceIndex]);
            points.push_back(Point(x, y, z));
        }
        catch (const std::invalid_argument& e)
        {
            // std::cerr << "Invalid argument: " << e.what() << std::endl;
            std::cerr << "Skipping first line with column names." << std::endl;
        }

        //std::cout << "done reading!" << std::endl;
    }
    // The points vector should/will have ~1.2M points to be used with the kMeansClustering function
    std::cout << "done reading!" << std::endl;
    return points;
}



int main() {
    std::vector<Point> data = readcsv();
    std::vector<Point>* points = &data;

    int n = points->size();
    int numPoints = n;
    std::cout << n << std::endl;
    // Randomly initialise centroids
    // The index of the centroid within the centroids vector represents the cluster label.
    std::vector<Point> centroids_data;
    srand(100);
    for (int i = 0; i < k; ++i)
    {
        centroids_data.push_back(points->at(rand() % n));
    }
    std::vector<Point>* centroids = &centroids_data;

    // Allocate GPU memory
    Point* d_points;
    cudaMalloc((void**)&d_points, sizeof(Point) * n);
    cudaMemcpy(d_points, points->data(), sizeof(Point) * n, cudaMemcpyHostToDevice);

    Point* d_centroids;
    cudaMalloc((void**)&d_centroids, sizeof(Point) * k);
    cudaMemcpy(d_centroids, centroids->data(), sizeof(Point) * k, cudaMemcpyHostToDevice);

    int* d_nPoints;
    cudaMalloc((void**)&d_nPoints, sizeof(int) * k);
    cudaMemset(d_nPoints, 0, sizeof(int) * k);

    double* d_sumX;
    cudaMalloc((void**)&d_sumX, sizeof(double) * k);
    cudaMemset(d_sumX, 0, sizeof(double) * k);

    double* d_sumY;
    cudaMalloc((void**)&d_sumY, sizeof(double) * k);
    cudaMemset(d_sumY, 0, sizeof(double) * k);

    double* d_sumZ;
    cudaMalloc((void**)&d_sumZ, sizeof(double) * k);
    cudaMemset(d_sumZ, 0, sizeof(double) * k);

    
    std::cout << "Running algorithm for " << epochs << " epochs" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    // Run k-means clustering on GPU
    for (int i = 0; i < epochs; ++i) {

        cudaDeviceSynchronize();
        // Assign points to clusters
        assignPointsToClusterstiled<<<ceil(1204025.0 / TILE_WIDTH), TILE_WIDTH>>>(d_points, numPoints, d_centroids, k, std::numeric_limits<double>::max());

        // Synchronize to ensure the previous kernel is finished
        cudaDeviceSynchronize();
        //std::cout << "part1" << std::endl;

        // Compute new centroids
        cudaMemset(d_nPoints, 0, sizeof(int) * k);
        cudaMemset(d_sumX, 0, sizeof(double) * k);
        cudaMemset(d_sumY, 0, sizeof(double) * k);
        cudaMemset(d_sumZ, 0, sizeof(double) * k);

        computeNewCentroids<<<1,k>>>(d_points, numPoints, d_centroids, k, d_nPoints, d_sumX, d_sumY, d_sumZ);

        // Synchronize to ensure the previous kernel is finished
        cudaDeviceSynchronize();

        //std::cout << "part2" << std::endl;

        updateNewCentroids<<<k, 1>>>(d_centroids,d_nPoints, d_sumX, d_sumY, d_sumZ);

        cudaDeviceSynchronize();
    }
    cudaMemcpy(points->data(), d_points, sizeof(Point) * n, cudaMemcpyDeviceToHost);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    long time = elapsed_time.count();
    double secs = ((double)time) / (1000 * 1000);
    std::cout << "Duration : " << secs << std::endl;
    // Write to csv
    std::cout << "Writing to CSV" << std::endl;
    std::ofstream myfile;
    myfile.open("output_shared_gpu_tiling.csv");
    myfile << "x,y,z,c" << std::endl;
    for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
    {
        myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster << std::endl;
    }
    myfile.close();

    // Free GPU memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_nPoints);
    cudaFree(d_sumX);
    cudaFree(d_sumY);
    cudaFree(d_sumZ);
    std::cout << "Finished successfully" << std::endl;

    return 0;
}
