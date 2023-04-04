
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
std::map<std::string, int> Timer::flops_;

  Timer::Timer(std::string label, int flops)
  : label_(label)
  {
    t_start_ = omp_get_wtime();
    gflops_it_ = flops/1E9;
  }



  Timer::~Timer()
  {
    double t_end = omp_get_wtime();
    times_[label_] += t_end - t_start_;
    gflops_[label_] += gflops_it_;
    counts_[label_]++;
  }

void Timer::summarize(std::ostream& os)
{
  os << "==================== TIMER SUMMARY =========================================" << std::endl;
  os << "label               \tcalls     \ttotal time\tmean time \ttotal flops \tmean flops\tflops/time "<<std::endl;
  os << "----------------------------------------------" << std::endl;
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
    int gflop = gflops_[label];
    std::cout << std::setw(20) << label << "\t" << std::setw(10) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) << time/double(count) << "\t";
    std::cout << std::setw(10) << gflop  << "\t" << std::setw(10) << gflop/double(count) << "\t" << std::setw(10) << gflop/time  << std::endl;
  }
  os << "============================================================================" << std::endl;
}
