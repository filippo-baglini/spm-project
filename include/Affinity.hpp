#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace affinity {

// Read an affinity specification from an environment variable
// Example:
//   AFFINITY=0,2,4-10:2
//
// If the variable is missing or empty, return an empty string.
// The caller can interpret an empty string as "no affinity requested"
inline std::string get_affinity_from_env(const char* envname = "AFFINITY") {
    const char* p = std::getenv(envname);
    if (p && *p) return std::string(p);
    return {};
}

// Parse a CPU list string such as:
//   "0,2,4-10:2"  ->  {0,2,4,6,8,10}
//
// Supported forms:
//   n       single CPU id
//   a-b     inclusive range
//   a-b:s   inclusive range with step s
//
// This helper only parses the syntax. It does not check whether the CPUs are
// valid on the current machine or allowed by the current cpuset
inline std::vector<unsigned> parse_cpu_list(const std::string& slist) {
    auto trim = [](std::string& s) {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
            s.erase(s.begin());
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
            s.pop_back();
    };

    auto parse_uint = [](std::string_view v) {
        if (v.empty())
            throw std::invalid_argument("affinity: empty numeric token");

        size_t idx = 0;
        unsigned x = std::stoul(std::string(v), &idx, 0);
        if (idx != v.size())
            throw std::invalid_argument("affinity: bad number '" + std::string(v) + "'");
        return x;
    };

    std::vector<unsigned> cpus;
    if (slist.empty()) return cpus;

    std::stringstream ss(slist);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        trim(tok);
        if (tok.empty()) continue;

        const auto dash = tok.find('-');
        const auto col  = tok.find(':');

        if (dash == std::string::npos) {
            cpus.push_back(parse_uint(tok));
        } else {
            unsigned a = parse_uint(std::string_view(tok).substr(0, dash));
            unsigned b;
            unsigned s = 1;

            if (col == std::string::npos) {
                b = parse_uint(std::string_view(tok).substr(dash + 1));
            } else {
                b = parse_uint(std::string_view(tok).substr(dash + 1, col - (dash + 1)));
                s = parse_uint(std::string_view(tok).substr(col + 1));
                if (s == 0)
                    throw std::invalid_argument("affinity: step 0");
            }

            if (a > b) std::swap(a, b);

            for (unsigned c = a; c <= b; c += s)
                cpus.push_back(c);
        }
    }

    return cpus;
}

#if defined(__linux__)

// Pin the current thread to exactly one logical CPU
//
// This is a hard restriction, not just a hint
// The function first checks that the requested CPU is inside the thread/process
// current allowed cpuset. This is useful when running under taskset, cgroups,
// containers, or a batch scheduler
inline bool pin_thread_to_core(unsigned cpu) {
    if (cpu >= CPU_SETSIZE) {
        std::printf("affinity::pin_thread_to_core, CPU index (%u) >= CPU_SETSIZE\n", cpu);
        return false;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        std::printf("affinity::pin_thread_to_core, sched_getaffinity failed errno=%d\n", errno);
        return false;
    }

    if (!CPU_ISSET(cpu, &allowed)) {
        std::printf("affinity::pin_thread_to_core, CPU %u not allowed by current cpuset\n", cpu);
        return false;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);

    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        std::printf("affinity::pin_thread_to_core, pthread_setaffinity_np failed rc=%d\n", rc);
        return false;
    }

    return true;
}

// Interpret the input vector as a bitmap of allowed CPUs
//
// Example:
//   {1,0,1,0} means "allow CPU 0 and CPU 2"
//
// This is different from pin_thread_to_cpu_list(), where the vector stores
// actual CPU ids. Here the index is the CPU id and the value is 0/1.
inline bool pin_thread_to_cores(const std::vector<unsigned>& bitmask) {
    if (bitmask.empty()) {
        std::printf("affinity::pin_thread_to_cores, bitmask vector is empty\n");
        return false;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        std::printf("affinity::pin_thread_to_cores, sched_getaffinity failed errno=%d\n", errno);
        return false;
    }

    cpu_set_t set;
    CPU_ZERO(&set);

    bool any = false;
    for (std::size_t c = 0; c < bitmask.size(); ++c) {
        if (!bitmask[c]) continue;

        if (c >= CPU_SETSIZE) {
            std::printf("affinity::pin_thread_to_cores, CPU index (%zu) >= CPU_SETSIZE\n", c);
            return false;
        }

        if (!CPU_ISSET(static_cast<int>(c), &allowed)) {
            std::printf("affinity::pin_thread_to_cores, CPU %zu not allowed by current cpuset\n", c);
            return false;
        }

        CPU_SET(static_cast<int>(c), &set);
        any = true;
    }

    if (!any) {
        std::printf("affinity::pin_thread_to_cores, bitmap selects no CPUs\n");
        return false;
    }

    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        std::printf("affinity::pin_thread_to_cores, pthread_setaffinity_np failed rc=%d\n", rc);
        return false;
    }

    return true;
}

// Interpret the input vector as a list of CPU ids
//
// Example:
//   {0,2,4} means "allow execution on logical CPUs 0, 2, and 4"
//
// This is the representation that is usually most convenient in experiments.
inline bool pin_thread_to_cpu_list(const std::vector<unsigned>& cpus) {
    if (cpus.empty()) {
        std::printf("affinity::pin_thread_to_cpu_list, empty list\n");
        return false;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        std::printf("affinity::pin_thread_to_cpu_list, sched_getaffinity failed errno=%d\n", errno);
        return false;
    }

    cpu_set_t set;
    CPU_ZERO(&set);

    for (unsigned c : cpus) {
        if (c >= CPU_SETSIZE) {
            std::printf("affinity::pin_thread_to_cpu_list, CPU index (%u) >= CPU_SETSIZE\n", c);
            return false;
        }

        if (!CPU_ISSET(c, &allowed)) {
            std::printf("affinity::pin_thread_to_cpu_list, CPU %u not allowed by current cpuset\n", c);
            return false;
        }

        CPU_SET(static_cast<int>(c), &set);
    }

    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        std::printf("affinity::pin_thread_to_cpu_list, pthread_setaffinity_np failed rc=%d\n", rc);
        return false;
    }

    return true;
}

// Query the affinity mask of the current thread and return the list of logical
// CPUs on which the thread is currently allowed to run.
inline bool get_thread_affinity(std::vector<unsigned>& out) {
    long nproc = ::sysconf(_SC_NPROCESSORS_CONF);
    if (nproc <= 0) nproc = CPU_SETSIZE;

    const size_t setsize = CPU_ALLOC_SIZE(nproc);
    cpu_set_t* set = CPU_ALLOC(nproc);
    if (!set) {
        std::printf("affinity::get_thread_affinity, CPU_ALLOC failed errno=%d\n", errno);
        return false;
    }

    CPU_ZERO_S(setsize, set);

    const int rc = ::pthread_getaffinity_np(::pthread_self(), setsize, set);
    if (rc != 0) {
        CPU_FREE(set);
        std::printf("affinity::get_thread_affinity, pthread_getaffinity_np failed rc=%d\n", rc);
        return false;
    }

    out.clear();
    out.reserve(static_cast<std::size_t>(nproc));
    for (int cpu = 0; cpu < nproc; ++cpu) {
        if (CPU_ISSET_S(cpu, setsize, set))
            out.push_back(static_cast<unsigned>(cpu));
    }

    CPU_FREE(set);
    return true;
}

// Return the logical CPU on which the current thread is executing right now
//
// Important: this is only an instantaneous observation. Without affinity, the
// scheduler may migrate the thread later to a different CPU
inline unsigned get_current_cpu() {
    const int cpu = ::sched_getcpu();
    if (cpu == -1) {
        std::printf("affinity::get_current_cpu, sched_getcpu failed errno=%d\n", errno);
        return static_cast<unsigned>(-1);
    }
    return static_cast<unsigned>(cpu);
}

#else
#error "thread affinity support only for Linux OS"
#endif

} // namespace affinity
