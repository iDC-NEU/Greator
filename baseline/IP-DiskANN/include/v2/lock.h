#pragma once
#include <atomic>
#include <thread>
#include <cstdlib>
#include <climits>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <assert.h>

class CASRWLock {
 public:
  CASRWLock() : lock(0) {
  }
  void ReadLock() {
    uint64_t i, n;
    uint64_t old_readers;
    for (;;) {
      old_readers = lock;
      if (old_readers != WLOCK &&
          __sync_bool_compare_and_swap(&lock, old_readers, old_readers + 1)) {
        return;
      }
      for (n = 1; n < SPIN; n <<= 1) {
        for (i = 0; i < n; i++) {
          __asm__("pause");
        }
        old_readers = lock;
        if (old_readers != WLOCK &&
            __sync_bool_compare_and_swap(&lock, old_readers, old_readers + 1)) {
          return;
        }
      }
      sched_yield();
    }
  }
  void WriteLock() {
    uint64_t i, n;
    for (;;) {
      if (lock == 0 && __sync_bool_compare_and_swap(&lock, 0, WLOCK)) {
        return;
      }
      for (n = 1; n < SPIN; n <<= 1) {
        for (i = 0; i < n; i++) {
          std::cout << "lock" << std::endl;
          __asm__("pause");
        }
        if (lock == 0 && __sync_bool_compare_and_swap(&lock, 0, WLOCK)) {
          return;
        }
      }
      sched_yield();
    }
  }
  void ReadUnLock() {
    uint64_t old_readers;
    old_readers = lock;
    if (old_readers == WLOCK) {
      lock = 0;
      return;
    }
    for (;;) {
      if (__sync_bool_compare_and_swap(&lock, old_readers, old_readers - 1)) {
        return;
      }
      old_readers = lock;
    }
  }
  void WriteUnlock() {
    uint64_t old_readers;
    old_readers = lock;
    if (old_readers == WLOCK) {
      lock = 0;
      return;
    }
    for (;;) {
      if (__sync_bool_compare_and_swap(&lock, old_readers, old_readers - 1)) {
        return;
      }
      old_readers = lock;
    }
  }

 private:
  static const uint64_t SPIN = 2048;
  static const uint64_t WLOCK = ((unsigned long) -1);
  uint64_t              lock;
};
