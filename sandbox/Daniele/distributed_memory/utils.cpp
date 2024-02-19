#include "utils.hpp"

bool utils::read_matrix_from_file(const char * filename, double ** matrix_out, size_t &num_rows_out, size_t &num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    num_rows_out = num_rows;
    num_cols_out = num_cols;

    fclose(file);

    return true;
}

bool utils::read_vector_from_file(const char * filename, double ** vector_out, size_t &length)
{
    double * vector;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&length, sizeof(size_t), 1, file);
    vector = new double[length];
    fread(vector, sizeof(double), length, file);

    *vector_out = vector;

    fclose(file);

    return true;
}

bool utils::read_matrix_rows(const char * filename, double ** matrix_out, size_t starting_row_num, size_t num_rows_to_read, size_t &num_cols)
{
    double * matrix;
    size_t num_rows;
    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "read_matrix_rows: Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    
    assert(starting_row_num + num_rows_to_read <= num_rows);

    matrix = new double[num_rows_to_read * num_cols];

    
    size_t offset = starting_row_num * num_cols + 2; 
    if (fseek(file, sizeof(double)*offset, SEEK_SET) != 0) {
        fprintf(stderr, "read_matrix_rows: Error setting file position");
        return false;
    }

    fread(matrix, sizeof(double), num_rows_to_read * num_cols, file);

    *matrix_out = matrix;

    fclose(file);

    return true;
}



bool utils::read_matrix_dims(const char * filename, size_t &num_rows_out, size_t &num_cols_out)
{

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "read_matrix_dims: Cannot open output file\n");
        return false;
    }

    fread(&num_rows_out, sizeof(size_t), 1, file);
    fread(&num_cols_out, sizeof(size_t), 1, file);


    fclose(file);

    return true;
}

void utils::print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file )
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}
