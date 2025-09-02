#ifndef STANDALONE_RAN2_H
#define STANDALONE_RAN2_H

#include <cstdint>
#include <cmath>

class Ran2
{
   public:
      Ran2(std::int64_t seed = -2) noexcept : _idum(seed)
      {
         _rn_is_set = false;
         _init();
      }

      void seed(std::int64_t seed_v) noexcept
      {
         _idum      = seed_v;
         _rn_is_set = false;
         _init();
      }

      [[nodiscard]] double uniform() noexcept
      {
         std::int64_t iy = _step_iy();
         double temp     = AM * static_cast<double>(iy);
         return (temp > RNMX) ? RNMX : temp;
      }

      [[nodiscard]] double uniform(double l, double m) noexcept
      {
         return (m - l) * uniform() + l;
      }

      [[nodiscard]] double normal() noexcept
      {
         double fac, rsq, v1, v2;
         if (!_rn_is_set) {
            do {
               v1  = 2.0 * uniform() - 1.0;
               v2  = 2.0 * uniform() - 1.0;
               rsq = v1 * v1 + v2 * v2;
            } while (rsq >= 1.0 || rsq == 0.0);
            fac = sqrt(-2.0 * log(rsq) / rsq);

            _rn_gset   = v1 * fac;
            _rn_is_set = true;
            return v2 * fac;
         } else {
            _rn_is_set = false;
            return _rn_gset;
         }
      }

      [[nodiscard]] double normal(double mu, double sigma) noexcept
      {
         return mu + sigma * normal();
      }

      [[nodiscard]] std::uint32_t uniform_uint(std::uint32_t n) noexcept
      {
         return static_cast<std::uint32_t>(std::max(0., std::floor(n * uniform())));
      }

      [[nodiscard]] double cauchy() noexcept
      {
         return tan(M_PI * (uniform() - 0.5));
      }

      [[nodiscard]] double cauchy(double m, double s) noexcept
      {
         return m + s * cauchy();
      }

      void discard(std::uint32_t n) noexcept
      {
         while (n--) {
            (void)_step_iy();
         }
      }

      // For interface with std library
      using result_type = std::uint32_t;
      static constexpr result_type min() noexcept
      {
         return 1u;
      }
      static constexpr result_type max() noexcept
      {
         return static_cast<result_type>(IMM1);
      }
      result_type operator()() noexcept
      {
         return static_cast<result_type>(_step_iy());
      }

   private:
      // from ran2 define macros
      static constexpr std::int64_t IM1  = 2147483563LL;
      static constexpr std::int64_t IM2  = 2147483399LL;
      static constexpr double AM         = (1.0 / (double)IM1);
      static constexpr std::int64_t IMM1 = (IM1 - 1);
      static constexpr std::int64_t IA1  = 40014LL;
      static constexpr std::int64_t IA2  = 40692LL;
      static constexpr std::int64_t IQ1  = 53668LL;
      static constexpr std::int64_t IQ2  = 52774LL;
      static constexpr std::int64_t IR1  = 12211LL;
      static constexpr std::int64_t IR2  = 3791LL;
      static constexpr int NTAB          = 32;
      static constexpr std::int64_t NDIV = (1 + IMM1 / NTAB);
      static constexpr double EPS        = 1.2e-7;
      static constexpr double RNMX       = (1.0 - EPS);

      // state
      std::int64_t _idum    = -2;
      std::int64_t _idum2   = 123456789LL;
      std::int64_t iy       = 0LL;
      std::int64_t iv[NTAB] = {};

      // For random normal
      bool _rn_is_set = false;
      double _rn_gset = 0.;

      // Methods
      void _init() noexcept
      {

         if (-(_idum) < 1) _idum = 1;
         else _idum = -(_idum);
         _idum2 = (_idum);
         for (int j = NTAB + 7; j >= 0; j--) {
            std::int64_t k = (_idum) / IQ1;
            _idum          = IA1 * (_idum - k * IQ1) - k * IR1;
            if (_idum < 0) _idum += IM1;
            if (j < NTAB) iv[j] = _idum;
         }
         iy = iv[0];
      }

      [[nodiscard]] std::int64_t _step_iy() noexcept
      {
         // _init();
         std::int64_t k = (_idum) / IQ1;
         _idum          = IA1 * (_idum - k * IQ1) - k * IR1;
         if (_idum < 0) _idum += IM1;
         k      = _idum2 / IQ2;
         _idum2 = IA2 * (_idum2 - k * IQ2) - k * IR2;
         if (_idum2 < 0) _idum2 += IM2;
         int j = iy / NDIV;
         iy    = iv[j] - _idum2;
         iv[j] = _idum;
         if (iy < 1) iy += IMM1;

         return iy;
      }
};

#endif // STANDALONE_RAN2_H