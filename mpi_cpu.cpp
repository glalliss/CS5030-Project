// Adapted from https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
// Dataset from https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs

/*
seg fault searching:
- Type is correct
- My data count is accurate
- The sendcounts add up to the total data count
- Points.data has no problem accessing all items (tested with for loop)
- Examples online use vector.data the same way I do
*/
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

void createPointType(MPI_Datatype *type) {
    MPI_Datatype types[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE};
    int blocklengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint offsets[5];

    // Get the offsets of each member in the struct
    offsets[0] = offsetof(Point, x);
    offsets[1] = offsetof(Point, y);
    offsets[2] = offsetof(Point, z);
    offsets[3] = offsetof(Point, cluster);
    offsets[4] = offsetof(Point, minDist);

    // Create the MPI datatype for Point
    MPI_Type_create_struct(5, blocklengths, offsets, types, type);

    if (MPI_Type_commit(type) != MPI_SUCCESS){
        printf("Type creation failed\n");
        MPI_Abort( MPI_COMM_WORLD , 1);
    }
}



// Reads in the data.csv file into a vector of points and return vector of points
std::vector<Point> readcsv()
{
    std::vector<Point> points;
    //std::ifstream file("tracks_features.csv");
    std::ifstream file("tracks_features.csv");
    std::string line;
    int danceabilityIndex = 9;
    int energyIndex = 10;
    int valenceIndex = 18;
    int count = 0;
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
            count++;
        }
        catch (const std::invalid_argument& e)
        {
            // std::cerr << "Invalid argument: " << e.what() << std::endl;
            std::cerr << "Skipping first line with column names." << std::endl;
        }
    }
    // The points vector should/will have ~1.2M points to be used with the kMeansClustering function
    printf("finished reading. num data points is %d.\n", count);
    return points;
}




/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
std::vector<Point> kMeansClustering(std::vector<Point>* points, int epochs, int k, int rank, MPI_Comm comm, MPI_Datatype mpi_point)
{
    int n = points->size();
    // Randomly initialise centroids
    // The index of the centroid within the centroids vector represents the cluster label.
    //Point* centroid_array = (Point*)malloc(k*sizeof(Point));
    srand(69);
    //std::vector<Point> centroids;
    Point *centroid_array = (Point*)malloc(k*sizeof(Point));
    if (rank == 0){
        for (int i = 0; i < k; ++i)
        {
            centroid_array[i] = points->at(rand() % n);
            //centroids.push_back(points->at(rand() % n));
        }
    }
    MPI_Bcast( centroid_array , k , mpi_point , 0 , comm);
    // Run algorithm however many epochs specified
    for (int i = 0; i < epochs; ++i)
    {
        std::vector<Point> centroids(centroid_array, centroid_array + k);        
        if (rank == 0){
            printf("Starting epoch %d.\n", i);
        }
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
            //printf("%d done with cluster %d\n", rank, clusterId);
        }
        // Create vectors to keep track of data needed to compute means

        int nPoints[k];
        int nPoints_global[k];
        double sumX[k];
        double sumY[k];
        double sumZ[k];
        double sumX_global[k];
        double sumY_global[k];
        double sumZ_global[k];

        // Initialize arrays
        for (int j = 0; j < k; ++j)
        {
            nPoints[j] = 0;
            nPoints_global[j] = 0;
            sumX_global[j] = 0.0;
            sumY_global[j] = 0.0;
            sumZ_global[j] = 0.0;
            sumX[j] = 0.0;
            sumY[j] = 0.0;
            sumZ[j] = 0.0;
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

        // Use the arrays as needed

        

        //send counts and sums back to process 0
        MPI_Reduce(nPoints, nPoints_global, k, MPI_INT, MPI_SUM, 0, comm);
        MPI_Reduce(sumX, sumX_global, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(sumY, sumY_global, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(sumZ, sumZ_global, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        
        // Compute the new centroids with agregated data
        if(rank == 0){
            
            for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c)
            {

                int clusterId = c - begin(centroids);
                printf("x: %f ", sumX_global[clusterId]);
                c->x = sumX_global[clusterId] / nPoints_global[clusterId];
                c->y = sumY_global[clusterId] / nPoints_global[clusterId];
                c->z = sumZ_global[clusterId] / nPoints_global[clusterId];
            }
            for (int pt = 0; pt < k; pt++)
            {
                centroid_array[pt] = centroids[pt];
            }
        }

        //sent centroids back to all processes.
        centroids.clear();
        MPI_Bcast(centroid_array, k, mpi_point, 0, comm);
    }
    printf("Kmeans complete!");
    return *points;
    // Write to csv
}

void write_csv(std::vector<Point> *points){
    std::ofstream myfile;
    myfile.open("output.csv");
    myfile << "x,y,z,c" << std::endl;
    printf("writing...");
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
    std::vector<Point> all_points;
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
        sendcounts[i] = data_size + (i < NUM_DATA % comm_size ? 1 : 0);
        displs[i] = NUM_DATA - remaining;
        remaining -= sendcounts[i];
    }
    MPI_Datatype mpi_point;
    createPointType(&mpi_point);

    if(my_rank == 0){
        all_points = readcsv();
        printf("Done reading data\n");
    }
    
    
    std::vector<Point> my_points(sendcounts[my_rank]);
    MPI_Scatterv(all_points.data(), sendcounts, displs, mpi_point, my_points.data(), sendcounts[my_rank], mpi_point, 0, comm);
    // Run k-means with specified number of iterations/epochs and specified number of clusters(k)
    
    my_points = kMeansClustering(&my_points, 100, 5, my_rank, comm, mpi_point);
    MPI_Gatherv(my_points.data(), sendcounts[my_rank], mpi_point, all_points.data(), sendcounts, displs, mpi_point, 0, comm);
    printf("Gather successful %d", my_rank);
    if(my_rank == 0){
        write_csv(&all_points);
    }
    MPI_Finalize();
}
