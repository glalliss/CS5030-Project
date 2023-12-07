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
#include <iterator>
#include <omp.h>
#include <time.h>

struct Point
{
    double x, y, z; // coordinates
    int cluster;    // no default cluster
    double minDist; // default infinite distance to nearest cluster
    // Initialize a point
    Point() :
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
std::vector<Point> readcsv(/*int thread_count*/)
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
    // #pragma omp parallel for private(line) shared(points) num_threads(thread_count)
    for(int i = 0; i < DATA_SIZE; i++)
    {
        //I think we oon't need this critical section if we read the data 
        // using offsets, but I don't have time right now.
        // #pragma omp critical
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
            points[i-1]  = Point(x, y, z);
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

std::vector<Point> kMeansClustering(std::vector<Point>* points, int epochs, int k, int thread_count)
{
    // Randomly initialise centroids
    std::cout << "Randomly initializing centroids" << std::endl;
    // The index of the centroid within the centroids vector represents the cluster label.
    std::vector<Point> centroids;
    srand(100);
    int n = points->size();
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back((*points)[rand() % n]);
    }

    // Run algorithm however many epochs specified
    std::cout << "Running algorithm for " << epochs << " epochs" << std::endl;
    for (int i = 0; i < epochs; ++i)
    {
        // For each centroid, compute distance from centroid to each point and update point's minDist and cluster if necessary
        #pragma omp parallel num_threads(thread_count)
        {
            #pragma omp for 
            for (int cIndex = 0; cIndex < k; ++cIndex)
            {
                int clusterId = cIndex;
                //calculate distance for each point
                for (int itIndex = 0; itIndex < points->size(); ++itIndex)
                {
                    Point p = (*points)[itIndex];
                    double dist = centroids[clusterId].distance(p);
                    //update the distance and cluster id
                    #pragma omp critical
                    if (dist < p.minDist)
                    {
                        p.minDist = dist;
                        p.cluster = clusterId;
                    }
                    (*points)[itIndex] = p;
                }
            }
        }

        // Create vectors to keep track of data needed to compute means
        std::vector<int> nPoints(k, 0);
        std::vector<double> sumX(k, 0.0), sumY(k, 0.0), sumZ(k, 0.0);

        // Iterate over points to append data to centroids
        #pragma omp parallel num_threads(thread_count)
        {
            #pragma omp for
            for (int itIndex = 0; itIndex < points->size(); ++itIndex)
            {
                int clusterId = (*points)[itIndex].cluster;
                #pragma omp atomic
                nPoints[clusterId] += 1;
                #pragma omp atomic
                sumX[clusterId] += (*points)[itIndex].x;
                #pragma omp atomic
                sumY[clusterId] += (*points)[itIndex].y;
                #pragma omp atomic
                sumZ[clusterId] += (*points)[itIndex].z;
                (*points)[itIndex].minDist = std::numeric_limits<double>::max(); // reset distance
            }

            // Compute the new centroids
            #pragma omp for
            for (int cIndex = 0; cIndex < k; ++cIndex)
            {
                int clusterId = cIndex;
                centroids[clusterId].x = sumX[clusterId] / nPoints[clusterId];
                centroids[clusterId].y = sumY[clusterId] / nPoints[clusterId];
                centroids[clusterId].z = sumZ[clusterId] / nPoints[clusterId];
            }
        }
    }
    return *points;
}



void write_csv(std::vector<Point> *points){
    std::ofstream myfile;
    myfile.open("output_shared_cpu.csv");
    myfile << "x,y,z,c" << std::endl;
    printf("writing...");
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
    int thread_count = std::stoi(argv[1]);
    std::vector<Point> points = readcsv();
    // Run k-means with specified number of iterations/epochs and specified number of clusters(k)
        clock_t start_time = clock();
        kMeansClustering(&points, 100, 5, thread_count);
        clock_t end_time = clock();
        write_csv(&points);

        double iteration_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC / thread_count;
        printf("%f\n", iteration_time);    
    std::cout << "Finished successfully" << std::endl;
}
