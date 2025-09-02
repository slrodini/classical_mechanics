#ifndef _TIMER_HPP
#define _TIMER_HPP

#include <chrono>

namespace timer
{
using mark = std::chrono::time_point<std::chrono::high_resolution_clock>;

inline mark now()
{
   return std::chrono::high_resolution_clock::now();
}

inline double elapsed_ns(const mark &end, const mark &begin)
{
   return static_cast<double>(
       std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

inline double elapsed_us(const mark &end, const mark &begin)
{
   return 1.0e-3
        * static_cast<double>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

inline double elapsed_ms(const mark &end, const mark &begin)
{
   return 1.0e-6
        * static_cast<double>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

inline double elapsed_s(const mark &end, const mark &begin)
{
   return 1.0e-9
        * static_cast<double>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

} // namespace timer

#endif // _TIMER_HPP