#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
bool file_exists(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file)
    {
        fclose(file);
        return true;
    }
    return false;
}

void open_file_to_write(FILE **writer, const char *filename)
{
    if (!file_exists(filename))
    {
        *writer = fopen(filename, "wb");
    }
    else
    {
        *writer = fopen(filename, "rb+");
    }

    if (*writer == NULL)
    {
        fprintf(stderr, "Failed to open file %s for write because %s\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

uint64_t save_bin_test(const char *filename, int *id, size_t npts, size_t ndims, size_t offset)
{
    FILE *writer;
    open_file_to_write(&writer, filename);

    printf("Writing bin: %s\n", filename);
    fseek(writer, offset, SEEK_SET);
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    size_t bytes_written = npts * ndims * sizeof(int) + 2 * sizeof(uint32_t);

    fwrite(&npts_i32, sizeof(int), 1, writer);
    fwrite(&ndims_i32, sizeof(int), 1, writer);
    printf("bin: #pts = %zu, #dims = %zu, size = %zu B\n", npts, ndims, bytes_written);

    for (size_t i = 0; i < npts; i++)
    {
        fwrite(id + i, sizeof(int), 1, writer);
    }

    fclose(writer);
    printf("Finished writing bin.\n");
    return bytes_written;
}

uint64_t save_bin_test_1(const char *filename, int *id, size_t npts, size_t offset)
{
    FILE *writer;
    open_file_to_write(&writer, filename);

    printf("Writing bin: %s\n", filename);
    fseek(writer, offset, SEEK_SET);
    int npts_i32 = (uint32_t)npts;
    size_t bytes_written = npts * sizeof(uint32_t) + sizeof(uint32_t);
    fwrite(&npts_i32, sizeof(uint32_t), 1, writer);

    for (size_t i = 0; i < npts; i++)
    {
        fwrite(id + i, sizeof(uint32_t), 1, writer);
    }
    for (size_t i = 0; i < npts; i++)
    {
        fwrite(id + i, sizeof(uint32_t), 1, writer);
    }

    fclose(writer);
    printf("Finished writing bin.\n");
    return bytes_written;
}

int main(int argc, char **argv)
{
    char *filename = argv[1];
    int num = atoi(argv[2]);
    int *id = (int *)malloc(num * sizeof(int));
    for (int i = 0; i < num; i++)
    {
        id[i] = i;
    }

    save_bin_test(filename, id, num, 1, 0);
}