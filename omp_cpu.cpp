// Adapted from https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
// Dataset from https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs


/*
* This version of the program only has a shared memory implementation.
*/
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>


struct Point
{
    double x, y, z; // coordinates
    int cluster;    // no default cluster
    double minDist; // default infinite distance to nearest cluster
    // Initialize a point
    Point():
        x(0.0), y(0.0), z(0.0), cluster(-1), minDist(std::numeric_limits<double>::max()) {}
    Point(double x, double y, double z) :
        x(x), y(y), z(z), cluster(-1), minDist(std::numeric_limits<double>::max()) {}
    // Computes the (square) euclidean distance between this point and another
    double distance(Point p)
    {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z);
    }
};

// Reads in the data.csv file into a vector of points and return vector of points
std::vector<Point> readcsv(int num_threads)
{
    std::cout << "Reading CSV into points" << std::endl;
    int DATA_SIZE = 1204026; //This includes the heading line.
    std::vector<Point> points(DATA_SIZE - 1);
    std::ifstream file("tracks_features.csv");
    std::string line;
    int danceabilityIndex = 9;
    int energyIndex = 10;
    int valenceIndex = 18;

    //line 1 is column titles, lines 2-120426 are data points.
    #pragma omp parallel for private(line) shared(points) num_threads(num_threads)
    for(int i = 0; i < DATA_SIZE; i++)
    {
        //I think we not need this critical section if we read the data 
        // using offsets, but I don't have time right now.
        #pragma omp critical
        getline(file, line);

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
            points[i]  = Point(x, y, z);
        }
        catch (const std::invalid_argument& e)
        {
            // std::cerr << "Invalid argument: " << e.what() << std::endl;
            std::cerr << "Skipping first line with column names at index " << i << std::endl;
        }
    }
    std::cout << "Finished reading CSV into points" << std::endl;
    // The points vector should/will have ~1.2M points to be used with the kMeansClustering function
    return points;
}

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */



//Kernal function: takes a single point
void kMeansClustering(std::vector<Point>* points, int epochs, int k, int num_threads)
{
    // Randomly initialise centroids
    std::cout << "Randomly initializing centroids" << std::endl;
    // The index of the centroid within the centroids vector represents the cluster label.
    std::vector<Point> centroids;
    srand(time(0));
    int n = points->size();
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(points->at(rand() % n));
    }
    // Run algorithm however many epochs specified
    std::cout << "Running algorithm for " << epochs << " epochs" << std::endl;
    for (int i = 0; i < epochs; ++i)
    {
        // std::cerr << "Starting epoch: " << i << std::endl;
        // For each centroid, compute distance from centroid to each point and update point's minDist and cluster if necessary
        for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);
            //calculate distance for each point

            //This is not the best way to do it. There will be some overhead each epoch to spawn the threads.
            #pragma omp parallel for num_threads(num_threads) 
            for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
            {
                Point p = *it;
                double dist = c->distance(p);

                //update the distance and cluster id
                #pragma omp critical
                if (dist < p.minDist)
                {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
                *it = p;
            }
        }
        // Create vectors to keep track of data needed to compute means
        std::vector<int> nPoints;
        std::vector<double> sumX, sumY, sumZ;
        #pragma omp single
        for (int j = 0; j < k; ++j)
        {
            //this simply initializes the vectors to 0
            nPoints.push_back(0);
            sumX.push_back(0.0);
            sumY.push_back(0.0);
            sumZ.push_back(0.0);
        }
        // Iterate over points to append data to centroids

        #pragma omp for
        for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
        {
            int clusterId = it->cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += it->x;
            sumY[clusterId] += it->y;
            sumZ[clusterId] += it->z;
            it->minDist = std::numeric_limits<double>::max(); // reset distance
        }
        // Compute the new centroids
        #pragma omp for
        for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);
            c->x = sumX[clusterId] / nPoints[clusterId];
            c->y = sumY[clusterId] / nPoints[clusterId];
            c->z = sumZ[clusterId] / nPoints[clusterId];
        }
    }
    // Write to csv
    std::cout << "Writing to CSV" << std::endl;
    std::ofstream myfile;
    myfile.open("output_shared_cpu.csv");
    myfile << "x,y,z,c" << std::endl;
    for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
    {
        myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster << std::endl;
    }
    myfile.close();
}

int main(int argc, char* argv[])
{
    if (argc != 2){
        std::cout << "Usage: <number of threads>" << std::endl;
        return 0;
    }
    int num_threads = std::stoi(argv[1]);
    std::vector<Point> points = readcsv(num_threads);
    // Run k-means with specified number of iterations/epochs and specified number of clusters(k)
    kMeansClustering(&points, 500, 5, num_threads);
    std::cout << "Finished successfully" << std::endl;
}
