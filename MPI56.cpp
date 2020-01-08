/*******************************************************************************************
This algorithm is a parallel implementation of Strassen's matrix multiplication algorithm
using MPI.  This program MUST be called with 56 nodes in the command line.

The implementation will recursively use up to the 56 nodes if needed.  If more than one 
recursive call to the matrix multiplication algorithm is required, a single node will be
used for subsequent calls.

To compile/run in linux:
mpicxx -std=c++11 -g -Wall -o mpi56 MPI56.cpp
mpirun -np 56 ./mpi56 [dimension] [max integer]

*******************************************************************************************/
#include <iostream>
#include <time.h>
#include <iomanip>
#include <string>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
using namespace std;

void FillMatrix(double matrix[], int dimension, int maxInt);
void StrassenMult(double matrix1[], double matrix2[], double matrix3[], int dim);
void StrassenMultMPI(double matrix1[], double matrix2[], double matrix3[], 
                        int dim, int my_rank, int startNode);
void FillSubmatrices(double matrix[], int dim, double sub1[],
                     double sub2[], double sub3[], double sub4[],
                     int subDim);
void AddMatrices(double matrix1[], double matrix2[], double matrix3[], int dim);
void SubtractMatrices(double matrix1[], double matrix2[], double matrix3[], int dim);
void DisplayMatrix(double matrix[], int dim);
void FillWithQuads(double quad1[], double quad2[], double quad3[],
                   double quad4[], int subDim, double matrix[], int dim);


int main(int argc, char* argv[])
{
     int my_rank;                                    // node number
     int dim;                                        // row/col dim of input matrices
     int maxInt;                                     // number of possible element values
     double* firstMatrix = NULL;                     // first input matrix
     double* secondMatrix = NULL;                    // second input matrix
     double* resultMatrix = NULL;                    // matrix mult result
     double startTime;                               // start of matrix mult
     double endTime;                                 // end of matrix mult
     int startNode = 0;                              // first of 7 MPI nodes for 
                                                     // StrassenMultMPI call
     // Check for correct argument count
     if (argc != 3)
     {
         cerr << "Incorrect number of args!\n";
         return 1;
     }

     // Set matrix dimension and max element size
     dim = stoi(argv[1]);
     maxInt = stoi(argv[2]);

     // Check for valid dimension
     if ((log(dim) / log(2)) != (int)(log(dim) / log(2)))  // dim must be 2^x
     {
         cerr << "Invalid dimension!\n";
         return 1;
     }

     // Check for valid maxInt
     if (maxInt < 0)
     {
         cerr << "Invalid max integer!\n";
         return 1;
     }

     // Initialize matrices
     firstMatrix = new double[dim * dim];
     secondMatrix = new double[dim * dim];
     resultMatrix = new double[dim * dim];

    // Initialize random number generator
    srand(time(NULL));

    // Initialize MPI
    if (MPI_Init(&argc,&argv) != MPI_SUCCESS)
    {
        printf("MPI-INIT Failed\n");
        return 1;
    }

    // Get MPI rank
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0)
    {
        // Fill first matrix
        FillMatrix(firstMatrix, dim, maxInt);

        // Fill second matrix
        FillMatrix(secondMatrix, dim, maxInt);
    }

    // Broadcast the two matrices to all nodes
    MPI_Bcast(firstMatrix, dim*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(secondMatrix, dim*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // Multiply the two matrices using Strassen's algorithm
    MPI_Barrier(MPI_COMM_WORLD);                 // ensure all nodes start together
    startTime = MPI_Wtime();
    StrassenMultMPI(firstMatrix, secondMatrix, resultMatrix,
                    dim, my_rank, startNode);
    endTime = MPI_Wtime();

    // Display results
    if (my_rank == 0)
    {
        printf("\nMultiplication of dimension %d matrices took %f"
               " seconds\n\n", dim, endTime - startTime);

        /*printf("first matrix:\n");
        DisplayMatrix(firstMatrix, dim);
        printf("second matrix:\n");
        DisplayMatrix(secondMatrix, dim);
        printf("result:\n");
        DisplayMatrix(resultMatrix, dim);*/
    }

    // Finalize MPI
    MPI_Finalize();

    // Deallocate matrices
    delete [] firstMatrix;
    delete [] secondMatrix;
    delete [] resultMatrix;

    return 0;
}

// Fill a matrix of a specified dimension with random integers
void FillMatrix(double matrix[], int dim, int maxInt)
{
    if (maxInt == 0)
        for (int i=0; i<(dim*dim); i++)
            matrix[i] = 1;
    else
        for (int i=0; i<(dim*dim); i++)
            matrix[i] = rand() % maxInt;                  // 0 <= matrix element < maxInt
}

// Multiply two matrices using Strassen's algorithm
void StrassenMult(double matrix1[], double matrix2[], double matrix3[], int dim)
{
    // Check for matrices with 1 element
    if (dim == 1)
    {
        matrix3[0] = matrix1[0] * matrix2[0];          // only int multipication needed
        return;
    }

    if (dim == 2)
    {
        double p1, p2, p3, p4, p5, p6, p7;         // hold results of Strassen's 7 equations

        // Find 7 equation results for 2 x 2 matrices
        p1 = matrix1[0] * (matrix2[1] - matrix2[3]);   // a(f-h)
        p2 = (matrix1[0] + matrix1[1]) * matrix2[3];   // (a+b)h
        p3 = (matrix1[2] + matrix1[3]) * matrix2[0];   // (c+d)e
        p4 = matrix1[3] * (matrix2[2] - matrix2[0]);   // d(g-e)
        p5 = (matrix1[0] + matrix1[3]) *
             (matrix2[0] + matrix2[3]);                // (a+d)(e+h)
        p6 = (matrix1[1] - matrix1[3]) *
             (matrix2[2] + matrix2[3]);                // (b-d)(g+h)
        p7 = (matrix1[0] - matrix1[2]) *
             (matrix2[0] + matrix2[1]);                // (a-c)(e+f)

        // Fill result matrix3 based on p1-p7
        matrix3[0] = p5 + p4 - p2 + p6;
        matrix3[1] = p1 + p2;
        matrix3[2] = p3 + p4;
        matrix3[3] = p1 + p5 - p3 - p7;
    }
    else
    {
        int subDim = dim / 2;                        // dim for each quadrant of the matrices
        int numElements = subDim * subDim;           // number of elements in sub-matrices

        // Initialize sub-matrices
        double* a = new double[numElements];         // top, left quadrant of matrix1
        double* b = new double[numElements];         // top, right quadrant of matrix1
        double* c = new double[numElements];         // bottom, left quadrant of matrix1
        double* d = new double[numElements];         // bottom, right quadrant of matrix1
        double* e = new double[numElements];         // top, left quadrant of matrix2
        double* f = new double[numElements];         // top, right quadrant of matrix2
        double* g = new double[numElements];         // bottom, left quadrant of matrix2
        double* h = new double[numElements];         // bottom, right quadrant of matrix2
        double* result1 = new double[numElements];   // the result of a sub-matrix operation
        double* result2 = new double[numElements];   // the result of a sub-matrix operation
        double* m1 = new double[numElements];        // matrix with result of Strassen's eq. 1
        double* m2 = new double[numElements];        // matrix with result of Strassen's eq. 2
        double* m3 = new double[numElements];        // matrix with result of Strassen's eq. 3
        double* m4 = new double[numElements];        // matrix with result of Strassen's eq. 4
        double* m5 = new double[numElements];        // matrix with result of Strassen's eq. 5
        double* m6 = new double[numElements];        // matrix with result of Strassen's eq. 6
        double* m7 = new double[numElements];        // matrix with result of Strassen's eq. 7
        double* quad1 = new double[numElements];     // top, left quadrant of matrix3
        double* quad2 = new double[numElements];     // top, right quadrant of matrix3
        double* quad3 = new double[numElements];     // bottom, left quadrant of matrix3
        double* quad4 = new double[numElements];     // bottom, right quadrant of matrix3

        // Fill sub-matrices a-h from matrix1 and matrix2
        FillSubmatrices(matrix1, dim, a, b, c, d, subDim);
        FillSubmatrices(matrix2, dim, e, f, g, h, subDim);

        // Find matrices m1-m7 with results for equations 1-7
        SubtractMatrices(f, h, result1, subDim);            // f-h
        StrassenMult(a, result1, m1, subDim);               // a(f-h)

        AddMatrices(a, b, result1, subDim);                 // a+b
        StrassenMult(result1, h, m2, subDim);               // (a+b)h

        AddMatrices(c, d, result1, subDim);                  // c+d
        StrassenMult(result1, e, m3, subDim);                // (c+d)e

        SubtractMatrices(g, e, result1, subDim);             // g-e
        StrassenMult(d, result1, m4, subDim);                // d(g-e)

        AddMatrices(a, d, result1, subDim);                  // a+d
        AddMatrices(e, h, result2, subDim);                  // e+h
        StrassenMult(result1, result2, m5, subDim);          // (a+d)(e+h)

        SubtractMatrices(b, d, result1, subDim);             // b-d
        AddMatrices(g, h, result2, subDim);                  // g+h
        StrassenMult(result1, result2, m6, subDim);          // (b-d)(g+h)

        SubtractMatrices(a, c, result1, subDim);             // a-c
        AddMatrices(e, f, result2, subDim);                  // e+f
        StrassenMult(result1, result2, m7, subDim);          // (a-c)(e+f)

        // Determine quadrants of matrix3 based on m1-m7
        AddMatrices(m5, m4, result1, subDim);                // m5+m4
        SubtractMatrices(result1, m2, result2, subDim);      // m5+m4-m2
        AddMatrices(result2, m6, quad1, subDim);             // m5+m4-m2+m6

        AddMatrices(m1, m2, quad2, subDim);                  // m1+m2

        AddMatrices(m3, m4, quad3, subDim);                  // m3+m4

        AddMatrices(m1, m5, result1, subDim);                // m1+m5
        SubtractMatrices(result1, m3, result2, subDim);      // m1+m5-m3
        SubtractMatrices(result2, m7, quad4, subDim);        // m1+m5-m3-m7

        // Fill matrix3 from quadrants
        FillWithQuads(quad1, quad2, quad3, quad4, subDim, matrix3, dim);

        // Deallocate sub-matrices
        delete [] a;
        delete [] b;
        delete [] c;
        delete [] d;
        delete [] e;
        delete [] f;
        delete [] g;
        delete [] h;
        delete [] result1;
        delete [] result2;
        delete [] m1;
        delete [] m2;
        delete [] m3;
        delete [] m4;
        delete [] m5;
        delete [] m6;
        delete [] m7;
        delete [] quad1;
        delete [] quad2;
        delete [] quad3;
        delete [] quad4;
    }
}


// Multiply two matrices using Strassen's algorithm and MPI
void StrassenMultMPI(double matrix1[], double matrix2[], double matrix3[], 
                        int dim, int my_rank, int startNode)
{
    // Check for matrices with 1 element
    if (dim == 1 && my_rank == 0)
    {
        matrix3[0] = matrix1[0] * matrix2[0];      // only int multipication needed
    }

    if (dim == 2)
    {
        double p1, p2, p3, p4, p5, p6, p7;         // hold results of Strassen's 7 equations

        // Find 7 equation results for 2 x 2 matrices
        p1 =  matrix1[0] * (matrix2[1]  - matrix2[3]);  // a(f-h)
        p2 = (matrix1[0] +  matrix1[1]) * matrix2[3];   // (a+b)h
        p3 = (matrix1[2] +  matrix1[3]) * matrix2[0];   // (c+d)e
        p4 =  matrix1[3] * (matrix2[2]  - matrix2[0]);  // d(g-e)
        p5 = (matrix1[0] +  matrix1[3]) *
             (matrix2[0] + matrix2[3]);                 // (a+d)(e+h)
        p6 = (matrix1[1] -  matrix1[3]) *
             (matrix2[2] + matrix2[3]);                 // (b-d)(g+h)
        p7 = (matrix1[0] -  matrix1[2]) *
             (matrix2[0] + matrix2[1]);                 // (a-c)(e+f)

        // Fill result matrix3 based on p1-p7
        matrix3[0] = p5 + p4 - p2 + p6;
        matrix3[1] = p1 + p2;
        matrix3[2] = p3 + p4;
        matrix3[3] = p1 + p5 - p3 - p7;
    }
    if (dim > 2)
    {
        int subDim = dim / 2;                        // dim for each quadrant of the matrices
        int numElements = subDim * subDim;           // number of elements in sub-matrices

        // Initialize sub-matrices
        double* a = new double[numElements];         // top, left quadrant of matrix1
        double* b = new double[numElements];         // top, right quadrant of matrix1
        double* c = new double[numElements];         // bottom, left quadrant of matrix1
        double* d = new double[numElements];         // bottom, right quadrant of matrix1
        double* e = new double[numElements];         // top, left quadrant of matrix2
        double* f = new double[numElements];         // top, right quadrant of matrix2
        double* g = new double[numElements];         // bottom, left quadrant of matrix2
        double* h = new double[numElements];         // bottom, right quadrant of matrix2
        double* result1 = new double[numElements];   // the result of a sub-matrix operation
        double* result2 = new double[numElements];   // the result of a sub-matrix operation
        double* m1 = new double[numElements];        // matrix with result of Strassen's eq. 1
        double* m2 = new double[numElements];        // matrix with result of Strassen's eq. 2
        double* m3 = new double[numElements];        // matrix with result of Strassen's eq. 3
        double* m4 = new double[numElements];        // matrix with result of Strassen's eq. 4
        double* m5 = new double[numElements];        // matrix with result of Strassen's eq. 5
        double* m6 = new double[numElements];        // matrix with result of Strassen's eq. 6
        double* m7 = new double[numElements];        // matrix with result of Strassen's eq. 7
        double* quad1 = new double[numElements];     // top, left quadrant of matrix3
        double* quad2 = new double[numElements];     // top, right quadrant of matrix3
        double* quad3 = new double[numElements];     // bottom, left quadrant of matrix3
        double* quad4 = new double[numElements];     // bottom, right quadrant of matrix3

        // Fill sub-matrices a-h from matrix1 and matrix2
        FillSubmatrices(matrix1, dim, a, b, c, d, subDim);
        FillSubmatrices(matrix2, dim, e, f, g, h, subDim);

        // Find matrices m1-m7 with results for equations 1-7
        if (my_rank == startNode || 
            ((my_rank >= (7+(7*startNode))) &&        // >= next-level start node
             (my_rank <= (13+(7*startNode)))))        // <= next-level end node
        {
            if ((7+(7*startNode)) > 49)                // 49 is max next-level start node
            {
                SubtractMatrices(f, h, result1, subDim);            // f-h
                StrassenMult(a, result1, m1, subDim);               // a(f-h)
            }
            else
            {
                if (my_rank != startNode)    // only next-level nodes should do recursive call
                {
                    SubtractMatrices(f, h, result1, subDim);              // f-h
                    StrassenMultMPI(a, result1, m1, subDim, my_rank,      // a(f-h)
                                    (7+(7*startNode)));    
                }
                if (my_rank == (7+(7*startNode)))         
                    // Send m1 from next-level startNode to current startNode
                    MPI_Send(m1, subDim*subDim, MPI_DOUBLE, startNode,
                             0, MPI_COMM_WORLD);
                if (my_rank == startNode)
                    // Receive m1
                    MPI_Recv(m1, subDim*subDim, MPI_DOUBLE, 
                             (7+(7*startNode)), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        if (my_rank == (startNode+1) ||
            ((my_rank >= (7+(7*(startNode+1)))) &&        // >= next-level start node
             (my_rank <= (13+(7*(startNode+1))))))        // <= next-level end node
        {
            if ((7+(7*(startNode+1))) > 49)                    // 49 is max next-level start node
            {
                AddMatrices(a, b, result1, subDim);                 // a+b
                StrassenMult(result1, h, m2, subDim);               // (a+b)h
                // Send m2 from startNode+1 to current startNode
                MPI_Send(m2, subDim*subDim, MPI_DOUBLE, startNode, 0, MPI_COMM_WORLD);
            }
            else
            {
                if (my_rank != (startNode+1))    // only next-level nodes should do recursive call
                {
                    AddMatrices(a, b, result1, subDim);                 // a+b
                    StrassenMultMPI(result1, h, m2, subDim, my_rank,    // (a+b)h
                                    (7+(7*(startNode+1)))); 
                }
                if (my_rank == (7+(7*(startNode+1))))
                    // Send m2 from next-level startNode to current startNode
                    MPI_Send(m2, subDim*subDim, MPI_DOUBLE, startNode,
                             0, MPI_COMM_WORLD);
            }
        }


        if (my_rank == (startNode+2) ||
            ((my_rank >= (7+(7*(startNode+2)))) &&        // >= next-level start node
             (my_rank <= (13+(7*(startNode+2))))))        // <= next-level end node
        {
            if ((7+(7*(startNode+2))) > 49)                    // 49 is max next-level start node
            {
                AddMatrices(c, d, result1, subDim);                  // c+d
                StrassenMult(result1, e, m3, subDim);                // (c+d)e
                // Send m3 from startNode+2 to current startNode
                MPI_Send(m3, subDim*subDim, MPI_DOUBLE, startNode, 0, MPI_COMM_WORLD);
            }
            else
            {
                if (my_rank != (startNode+2))    // only next-level nodes should do recursive call
                {
                    AddMatrices(c, d, result1, subDim);                  // c+d
                    StrassenMultMPI(result1, e, m3, subDim, my_rank,     // (c+d)e
                                    (7+(7*(startNode+2))));
                }
                if (my_rank == (7+(7*(startNode+2))))
                    // Send m3 from next-level startNode to current startNode
                    MPI_Send(m3, subDim*subDim, MPI_DOUBLE, startNode,
                             0, MPI_COMM_WORLD);
            }
        }

        if (my_rank == (startNode+3) ||
            ((my_rank >= (7+(7*(startNode+3)))) &&        // >= next-level start node
             (my_rank <= (13+(7*(startNode+3))))))        // <= next-level end node
        {
            if ((7+(7*(startNode+3))) > 49)                    // 49 is max next-level start node
            {
                SubtractMatrices(g, e, result1, subDim);             // g-e
                StrassenMult(d, result1, m4, subDim);                // d(g-e)
                // Send m4 from startNode+3 to current startNode
                MPI_Send(m4, subDim*subDim, MPI_DOUBLE, startNode, 0, MPI_COMM_WORLD);
            }
            else
            {
                 if (my_rank != (startNode+3))    // only next-level nodes should do recursive call
                 {
                      SubtractMatrices(g, e, result1, subDim);             // g-e
                      StrassenMultMPI(d, result1, m4, subDim, my_rank,     // d(g-e)
                                      (7+(7*(startNode+3))));
                 }
                 if (my_rank == (7+(7*(startNode+3))))
                    // Send m4 from next-level startNode to current startNode
                    MPI_Send(m4, subDim*subDim, MPI_DOUBLE, startNode,
                             0, MPI_COMM_WORLD);
            }
        }

        if (my_rank == (startNode+4) ||
            ((my_rank >= (7+(7*(startNode+4)))) &&        // >= next-level start node
             (my_rank <= (13+(7*(startNode+4))))))        // <= next-level end node
        {
             if ((7+(7*(startNode+4))) > 49)                    // 49 is max next-level start node
             {
                AddMatrices(a, d, result1, subDim);                  // a+d
                AddMatrices(e, h, result2, subDim);                  // e+h
                StrassenMult(result1, result2, m5, subDim);          // (a+d)(e+h)
                // Send m5 from startNode+4 to current startNode
                MPI_Send(m5, subDim*subDim, MPI_DOUBLE, startNode, 0, MPI_COMM_WORLD);
             }
             else
             {
                 if (my_rank != (startNode+4))    // only next-level nodes should do recursive call
                 {
                     AddMatrices(a, d, result1, subDim);                     // a+d
                     AddMatrices(e, h, result2, subDim);                     // e+h
                     StrassenMultMPI(result1, result2, m5, subDim, my_rank,  // (a+d)(e+h)
                                     (7+(7*(startNode+4))));
                 }
                  if (my_rank == (7+(7*(startNode+4))))
                    // Send m5 from next-level startNode to current startNode
                    MPI_Send(m5, subDim*subDim, MPI_DOUBLE, startNode,
                             0, MPI_COMM_WORLD);
             }
        }

        if (my_rank == (startNode+5) ||
            ((my_rank >= (7+(7*(startNode+5)))) &&        // >= next-level start node
             (my_rank <= (13+(7*(startNode+5))))))        // <= next-level end node
        {
            if ((7+(7*(startNode+5))) > 49)                    // 49 is max next-level start node
            {
                SubtractMatrices(b, d, result1, subDim);             // b-d
                AddMatrices(g, h, result2, subDim);                  // g+h
                StrassenMult(result1, result2, m6, subDim);          // (b-d)(g+h)
                // Send m6 from startNode+5 to current startNode
                MPI_Send(m6, subDim*subDim, MPI_DOUBLE, startNode, 0, MPI_COMM_WORLD);
            }
            else
            {
                if (my_rank != (startNode+5))    // only next-level nodes should do recursive call
                {
                    SubtractMatrices(b, d, result1, subDim);               // b-d
                    AddMatrices(g, h, result2, subDim);                    // g+h
                    StrassenMultMPI(result1, result2, m6, subDim, my_rank, // (b-d)(g+h)
                                    (7+(7*(startNode+5))));
                }
                if (my_rank == (7+(7*(startNode+5))))
                    // Send m6 from next-level startNode to current startNode
                    MPI_Send(m6, subDim*subDim, MPI_DOUBLE, startNode,
                             0, MPI_COMM_WORLD);
            }
        }

        if (my_rank == (startNode+6) ||
                 ((my_rank >= (7+(7*(startNode+6)))) &&        // >= next-level start node
                  (my_rank <= (13+(7*(startNode+6))))))        // <= next-level end node
        {
            if ((7+(7*(startNode+6))) > 49)                    // 49 is max next-level start node
            {
                SubtractMatrices(a, c, result1, subDim);             // a-c
                AddMatrices(e, f, result2, subDim);                  // e+f
                StrassenMult(result1, result2, m7, subDim);          // (a-c)(e+f)
                // Send m7 from startNode+6 to current startNode
                MPI_Send(m7, subDim*subDim, MPI_DOUBLE, startNode, 0, MPI_COMM_WORLD);
            }
            else
            {
                if (my_rank != (startNode+6))    // only next-level nodes should do recursive call
                {
                    SubtractMatrices(a, c, result1, subDim);                // a-c
                    AddMatrices(e, f, result2, subDim);                     // e+f
                    StrassenMultMPI(result1, result2, m7, subDim, my_rank,  // (a-c)(e+f)
                                    (7+(7*(startNode+6))));
                }
                if (my_rank == (7+(7*(startNode+6))))
                    // Send m7 from next-level startNode to current startNode
                    MPI_Send(m7, subDim*subDim, MPI_DOUBLE, startNode,
                             0, MPI_COMM_WORLD);
            }
        }

        if (my_rank == startNode)
        {
            // Receive m2-m7
            MPI_Recv(m2, subDim*subDim, MPI_DOUBLE, 
                     MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv(m3, subDim*subDim, MPI_DOUBLE, 
                     MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv(m4, subDim*subDim, MPI_DOUBLE, 
                     MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv(m5, subDim*subDim, MPI_DOUBLE, 
                     MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv(m6, subDim*subDim, MPI_DOUBLE, 
                     MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv(m7, subDim*subDim, MPI_DOUBLE, 
                     MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Determine quadrants of matrix3 based on m1-m7
            AddMatrices(m5, m4, result1, subDim);                // m5+m4
            SubtractMatrices(result1, m2, result2, subDim);      // m5+m4-m2
            AddMatrices(result2, m6, quad1, subDim);             // m5+m4-m2+m6

            AddMatrices(m1, m2, quad2, subDim);                  // m1+m2

            AddMatrices(m3, m4, quad3, subDim);                  // m3+m4

            AddMatrices(m1, m5, result1, subDim);                // m1+m5
            SubtractMatrices(result1, m3, result2, subDim);      // m1+m5-m3
            SubtractMatrices(result2, m7, quad4, subDim);        // m1+m5-m3-m7

            // Fill matrix3 from quadrants
            FillWithQuads(quad1, quad2, quad3, quad4, subDim, matrix3, dim);
        }

        // Deallocate sub-matrices
        delete [] a;
        delete [] b;
        delete [] c;
        delete [] d;
        delete [] e;
        delete [] f;
        delete [] g;
        delete [] h;
        delete [] result1;
        delete [] result2;
        delete [] m1;
        delete [] m2;
        delete [] m3;
        delete [] m4;
        delete [] m5;
        delete [] m6;
        delete [] m7;
        delete [] quad1;
        delete [] quad2;
        delete [] quad3;
        delete [] quad4;
    }
}

// Divide a matrix up by quadrant into 4 sub-matrices.  They are
// sub1, sub2, sub3, and sub4 starting in the top, left quadrant
// of the matrix, going left to right, and top to bottom.
void FillSubmatrices(double matrix[], int dim, double sub1[],
                     double sub2[], double sub3[], double sub4[],
                     int subDim)
{
    int index1 = 0;                                   // index of a sub-matrix 1 element
    int index2 = 0;                                   // index of a sub-matrix 2 element
    int index3 = 0;                                   // index of a sub-matrix 3 element
    int index4 = 0;                                   // index of a sub-matrix 4 element
    int matrixElement = 0;                            // index of a matrix element

    for (int row=0; row<subDim; row++)
        for (int col=0; col<dim; col++)
        {
            if (col < subDim)
            {
                // Set sub1 element from matrix quadrant 1
                sub1[index1] = matrix[matrixElement];
                index1++;
            }
            else
            {
                // Set sub2 element from matrix quadrant 2
                sub2[index2] = matrix[matrixElement];
                index2++;
            }
            matrixElement++;
        }

   for (int row=subDim; row<dim; row++)
       for (int col=0; col<dim; col++)
       {
           if (col < subDim)
           {
               // Set sub3 element from matrix quadrant 3
               sub3[index3] = matrix[matrixElement];
               index3++;
           }
           else
           {
               // Set sub4 element from matrix quadrant 4
               sub4[index4] = matrix[matrixElement];
               index4++;
           }
           matrixElement++;
       }
}

// Add two matrices together into a result matrix
void AddMatrices(double matrix1[], double matrix2[], double matrix3[], int dim)
{
    for (int i=0; i<(dim*dim); i++)
        matrix3[i] = matrix1[i] + matrix2[i];
}

// Subtract two matrices.  Subtract matrix2 from matrix1 and put the difference
// into a result matrix.
void SubtractMatrices(double matrix1[], double matrix2[], double matrix3[], int dim)
{
    for (int i=0; i<(dim*dim); i++)
        matrix3[i] = matrix1[i] - matrix2[i];
}

// Display a matrix
void DisplayMatrix(double matrix[], int dim)
{
    for (int i=0; i<(dim*dim); i++)
    {
        if (i % dim == 0)                             // return at end of matrix line
            cout << endl;
        cout << " " << matrix[i];
    }
    cout << endl << endl;
}

// Fill a matrix with elements from four separate quadrant sub-matrices
void FillWithQuads(double quad1[], double quad2[], double quad3[],
                   double quad4[], int subDim, double matrix[], int dim)
{
    int index1 = 0;                                   // index of a quad1 element
    int index2 = 0;                                   // index of a quad2 element
    int index3 = 0;                                   // index of a quad3 element
    int index4 = 0;                                   // index of a quad4 element
    int matrixElement = 0;                            // index of a matrix element

    for (int row=0; row<subDim; row++)
        for (int col=0; col<dim; col++)
        {
            if (col < subDim)
            {
                // Set matrix element from quad1
                matrix[matrixElement] = quad1[index1];
                index1++;
            }
            else
            {
                // Set matrix element from quad2
                matrix[matrixElement] = quad2[index2];
                index2++;
            }
            matrixElement++;
        }

    for (int row=subDim; row<dim; row++)
        for (int col=0; col<dim; col++)
        {
            if (col < subDim)
            {
                // Set matrix element from quad3
                matrix[matrixElement] = quad3[index3];
                index3++;
            }
            else
            {
                // Set matrix element from quad4
                matrix[matrixElement] = quad4[index4];
                index4++;
            }
            matrixElement++;
        }
}