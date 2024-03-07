#ifndef TIMER_HPP
#define TIMER_HPP

#include <vector>
#include <chrono>
#include <iostream>
#include <string>

namespace cgcore {

    /**
     * @brief This class is used to measure the execution time of different CG strategies.
     **/
    class Timer{

        public:

        using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
        using duration = std::chrono::duration<double>;

        Timer() = default;
        ~Timer() = default;

        void start();
        void stop();
        void reset();

        void print() const;
        void print(std::string) const;
        void print_last_formatted() const;
        std::string get_last_formatted() const;

        double get_last() const;
        double get_min() const;

        private: 
            time_point _current_start;
            std::vector<duration> _durations;
    };

}

#endif // TIMER_HPP