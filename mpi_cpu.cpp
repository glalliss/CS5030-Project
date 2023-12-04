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
#include <mpi.h>


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

MPI_Datatype createPointType() {
    MPI_Datatype mpiPointType;
    MPI_Datatype types[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    int blocklengths[4] = {1, 1, 1, 1};
    MPI_Aint offsets[4];

    // Get the offsets of each member in the struct
    offsets[0] = offsetof(Point, x);
    offsets[1] = offsetof(Point, y);
    offsets[2] = offsetof(Point, z);
    offsets[3] = offsetof(Point, cluster);

    // Create the MPI datatype for Point
    MPI_Type_create_struct(4, blocklengths, offsets, types, &mpiPointType);
    MPI_Type_commit(&mpiPointType);

    return mpiPointType;
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
    }
    // The points vector should/will have ~1.2M points to be used with the kMeansClustering function
    printf("finished writing\n");
    return points;
}




/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
std::vector<Point> kMeansClustering(std::vector<Point>* points, int epochs, int k, int rank, MPI_Comm comm)
{
    MPI_Datatype mpi_point = createPointType();
    int n = points->size();
    // Randomly initialise centroids
    // The index of the centroid within the centroids vector represents the cluster label.
    std::vector<Point> centroids;
    srand(time(0));
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(points->at(rand() % n));
    }
    // Run algorithm however many epochs specified
    for (int i = 0; i < epochs; ++i)
    {
        // For each centroid, compute distance from centroid to each point and update point's minDist and cluster if necessary
        for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c)
        {
            //This line computes the cluster that the points will all be compared to in the nested loop
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
        std::vector<int> nPoints, nPoints_global;
        std::vector<double> sumX, sumY, sumZ, sumX_global, sumY_global, sumZ_global;
        for (int j = 0; j < k; ++j)
        {
            nPoints.push_back(0);
            nPoints_global.push_back(0);
            sumX_global.push_back(0.0);
            sumY_global.push_back(0.0);
            sumZ_global.push_back(0.0);
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

        //send counts and sums back to process 0
        MPI_Reduce(nPoints.data(), std::data(nPoints_global) ,k , MPI_INTEGER  , MPI_SUM  , 0 , comm);
        MPI_Reduce(sumX.data(), std::data(sumX_global) , k , MPI_INTEGER  , MPI_SUM  , 0 , comm);
        MPI_Reduce(sumY.data() , std::data(sumY_global) , k , MPI_INTEGER  , MPI_SUM  , 0 , comm);
        MPI_Reduce(sumZ.data(), std::data(sumZ_global) , k , MPI_INTEGER  , MPI_SUM  , 0 , comm);
        
        // Compute the new centroids with agregated data
        if(rank == 0){
            for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c)
            {
                int clusterId = c - begin(centroids);
                c->x = sumX_global[clusterId] / nPoints_global[clusterId];
                c->y = sumY_global[clusterId] / nPoints_global[clusterId];
                c->z = sumZ_global[clusterId] / nPoints_global[clusterId];
            }
        }

        //sent centroids back to all processes.
        MPI_Bcast(centroids.data(), k, mpi_point, 0, comm);
    }
    printf("Size: %ld\n", points->size());
    return *points;
    // Write to csv
}

void write_csv(std::vector<Point> *points){
    std::ofstream myfile;
    myfile.open("output.csv");
    myfile << "x,y,z,c" << std::endl;
    for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
    {
        myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster << std::endl;
    }
    myfile.close();
}



int main()
{
    //there are 1204025 points.
    int NUM_DATA = 1204025;
    int my_rank, comm_size;
    int *sendcounts, *displs;
    std::vector<Point> points;
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm , &my_rank);
    MPI_Comm_size(comm, &comm_size);
    sendcounts = (int*)malloc(comm_size*sizeof(int));
    displs = (int*)malloc(comm_size*sizeof(int));
    int data_size = NUM_DATA/comm_size;
    int remaining = NUM_DATA;

    //calculate number of data to send each process
    for (int i = 0; i < comm_size; ++i) {
        sendcounts[i] = data_size + (i + 1 < data_size % comm_size ? 1 : 0);
        displs[i] = NUM_DATA - remaining;
        remaining -= sendcounts[i];
    }

    if(my_rank == 0){
        points = readcsv();
        printf("Done reading data\n");
        for (int i = 0; i < comm_size; i++)
        {
            printf("%d count: %d\n", i, sendcounts[i]);
            printf("%d Displacement: %d\n", i, displs[i]);
        }
        
    }
    MPI_Barrier(comm);
    MPI_Datatype mpi_point = createPointType();
    printf("type created %d\n", my_rank);
    std::vector<Point> my_points(sendcounts[my_rank]);
    printf("%d vector size: %ld\n", my_rank, my_points.size());
    MPI_Scatterv(points.data(), sendcounts, displs, mpi_point, my_points.data(), sendcounts[my_rank], mpi_point, 0, comm);
    // Run k-means with specified number of iterations/epochs and specified number of clusters(k)
    my_points = kMeansClustering(&my_points, 5000, 5, my_rank, comm);
    MPI_Gatherv(my_points.data(), sendcounts[my_rank], mpi_point, points.data(), sendcounts, displs, mpi_point, 0, comm);
    if(my_rank == 0){
        write_csv(&points);
    }
    MPI_Finalize();
}
