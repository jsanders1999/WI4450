
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
    flops_it_ = flops;
  }



  Timer::~Timer()
  {
    double t_end = omp_get_wtime();
    times_[label_] += t_end - t_start_;
    flops_[label_] += flops_;
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
    int flop = flops_[label];
    std::cout << std::setw(20) << label << "\t" << std::setw(10) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) << time/double(count) << "\t" <<;
    std::cout << std::setw(10) << flop  << "\t" << std::setw(10) << flop/double(count) << "\t" << std::setw(10) << flop/time  << std::endl;
  }
  os << "============================================================================" << std::endl;
}
