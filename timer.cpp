
#include "timer.hpp"

#include <iomanip>

// for timing routine
#include <omp.h>

#include <iostream>
#include <iomanip>

// static members of a class must be defined
// somewhere in an object file, otherwise you
// will get linker errors (undefined reference)
std::map<std::string, int> Timer::counts_;
std::map<std::string, double> Timer::times_;
std::map<std::string, double> Timer::gflops_;
std::map<std::string, double> Timer::gstream_;

  Timer::Timer(std::string label, double flops, double datastream)
  : label_(label)
  {
    t_start_ = omp_get_wtime();
    gflops_it_ = double(flops)/1E9;
    gstream_it_ = double(datastream)/1E9;
  }



  Timer::~Timer()
  {
    double t_end = omp_get_wtime();
    times_[label_] += t_end - t_start_;
    gflops_[label_] += gflops_it_;
    gstream_[label_] += gstream_it_;
    counts_[label_]++;
  }

void Timer::summarize(std::ostream& os)
{
  os << "==================== TIMER SUMMARY =========================================" << std::endl;
  os << "label               \tcalls     \ttotal time \tmean time \ttotal Gflops \tmean Gflops  \tGflops/s\ttotal datastream\tmean datastream\tbandwidth Gbyte/s\t\t I "<<std::endl;
  os << "----------------------------------------------" << std::endl;
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
    double gflop = gflops_[label];
    double gstream = gstream_[label];
    std::cout << std::setw(20) << label << "\t" << std::setw(10) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) << time/double(count) << "\t";
    std::cout << std::setw(10) << gflop  << "\t" << std::setw(10) << gflop/double(count) << "\t" << std::setw(10) << gflop/time << "\t";
    std::cout << std::setw(10) << gstream  << "\t" << std::setw(10) << gstream/double(count) << "\t" << std::setw(10) << gstream/time   << "\t" << std::setw(10) << gflops/gstream std::endl;
  }
  os << "============================================================================" << std::endl;
}
