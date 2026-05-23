#pragma once

#include <chrono>
#include <string>
#include <iostream>

namespace hpc {

/**
 * @class Timer
 * @brief A high-resolution timer utility for profiling C++ execution times.
 * 
 * Under the hood, it uses std::chrono::high_resolution_clock, which maps to
 * highly accurate system counters (such as mach_absolute_time on macOS).
 */
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_stop;
    bool m_running;

public:
    /**
     * @brief Construct a new Timer and optionally start it immediately.
     */
    Timer(bool start_immediately = false) : m_running(false) {
        if (start_immediately) {
            start();
        }
    }

    /**
     * @brief Start the timer.
     */
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
        m_running = true;
    }

    /**
     * @brief Stop the timer.
     */
    void stop() {
        if (m_running) {
            m_stop = std::chrono::high_resolution_clock::now();
            m_running = false;
        }
    }

    /**
     * @brief Get elapsed time in seconds.
     */
    double elapsed_seconds() const {
        auto end = m_running ? std::chrono::high_resolution_clock::now() : m_stop;
        std::chrono::duration<double> diff = end - m_start;
        return diff.count();
    }

    /**
     * @brief Get elapsed time in milliseconds.
     */
    double elapsed_milliseconds() const {
        return elapsed_seconds() * 1000.0;
    }

    /**
     * @brief Get elapsed time in microseconds.
     */
    double elapsed_microseconds() const {
        return elapsed_seconds() * 1000000.0;
    }

    /**
     * @brief Print the elapsed time directly with a custom prefix message.
     */
    void print(const std::string& label = "Elapsed time") const {
        std::cout << label << ": " << elapsed_milliseconds() << " ms (" 
                  << elapsed_seconds() << " seconds)" << std::endl;
    }
};

} // namespace hpc
