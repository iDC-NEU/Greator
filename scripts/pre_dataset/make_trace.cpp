#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <iostream>

bool file_exists(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file) {
    fclose(file);
    return true;
  }
  return false;
}

void open_file_to_write(FILE **writer, const char *filename) {
  if (!file_exists(filename)) {
    *writer = fopen(filename, "wb");
  } else {
    *writer = fopen(filename, "rb+");
  }

  if (*writer == NULL) {
    fprintf(stderr, "Failed to open file %s for write because %s\n", filename,
            strerror(errno));
    exit(EXIT_FAILURE);
  }
}

uint64_t save_bin_test(const char *filename, int *id, size_t npts, size_t ndims,
                       size_t offset) {
  FILE *writer;
  open_file_to_write(&writer, filename);

  printf("Writing bin: %s\n", filename);
  fseek(writer, offset, SEEK_SET);
  int    npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  size_t bytes_written = npts * ndims * sizeof(int) + 2 * sizeof(uint32_t);

  fwrite(&npts_i32, sizeof(int), 1, writer);
  fwrite(&ndims_i32, sizeof(int), 1, writer);
  printf("bin: #pts = %zu, #dims = %zu, size = %zu B\n", npts, ndims,
         bytes_written);

  for (size_t i = 0; i < npts; i++) {
    fwrite(id + i, sizeof(int), 1, writer);
  }

  fclose(writer);
  printf("Finished writing bin.\n");
  return bytes_written;
}

uint64_t save_bin_test_1(const char *filename, int startid, int endid,
                         size_t npts, size_t offset) {
  FILE *writer;
  open_file_to_write(&writer, filename);

  printf("Writing bin: %s\n", filename);
  fseek(writer, offset, SEEK_SET);
  int    npts_i32 = (uint32_t) npts;
  size_t bytes_written = npts * sizeof(uint32_t) + sizeof(uint32_t);
  fwrite(&npts_i32, sizeof(uint32_t), 1, writer);

  std::cout << "delete_num: " << npts << std::endl;
  for (size_t i = 0; i < npts; i++) {
    int id = startid + i;
    fwrite(&id, sizeof(uint32_t), 1, writer);
  }
  std::cout << "delete_id write done!" << std::endl;
  std::cout << "insert_num: " << npts << std::endl;
  for (size_t i = 0; i < npts; i++) {
    int id = endid + i;
    fwrite(&id, sizeof(uint32_t), 1, writer);
  }
  std::cout << "insert_id write done!" << std::endl;
  fclose(writer);
  printf("Finished writing bin.\n");
  return bytes_written;
}

int main(int argc, char **argv) {
  char *filename = argv[1];
  int   npts = atoi(argv[2]);
  int   delta = atoi(argv[3]);
  int   step = atoi(argv[4]);

  for (int ii = 0; ii < step; ii++) {
    int  startid = 0 + ii * delta;
    int  endid = 950000 + ii * delta;
    char file[256];
    std::cout << "start: " << startid << " end: " << endid << std::endl;
    snprintf(file, sizeof(file), "%s%d", filename, ii);
    save_bin_test_1(file, startid, endid, delta, 0);
  }
  return 0;
}
