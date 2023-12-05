// Adapted from https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
// Dataset from https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

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
std::vector<Point> readcsv()
{
    std::cout << "Reading CSV into points" << std::endl;
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
void kMeansClustering(std::vector<Point>* points, int epochs, int k)
{
    // Randomly initialise centroids
    std::cout << "Randomly initializing centroids" << std::endl;
    // The index of the centroid within the centroids vector represents the cluster label.
    std::vector<Point> centroids;
    srand(100);
    int n = points->size();
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(points->at(rand() % n));
    }
    // Run algorithm however many epochs specified
    std::cout << "Running algorithm for " << epochs << " epochs" << std::endl;
    for (int i = 0; i < epochs; ++i)
    {
        // For each centroid, compute distance from centroid to each point and update point's minDist and cluster if necessary
        for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);
            for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
            {
                Point p = *it;
                double dist = c->distance(p);
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
        for (int j = 0; j < k; ++j)
        {
            nPoints.push_back(0);
            sumX.push_back(0.0);
            sumY.push_back(0.0);
            sumZ.push_back(0.0);
        }
        // Iterate over points to append data to centroids
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
    myfile.open("output_serial.csv");
    myfile << "x,y,z,c" << std::endl;
    for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
    {
        myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster << std::endl;
    }
    myfile.close();
}

int main()
{
    std::vector<Point> points = readcsv();
    // Run k-means with specified number of iterations/epochs and specified number of clusters(k)
    kMeansClustering(&points, 100, 5);
    std::cout << "Finished successfully" << std::endl;
}
