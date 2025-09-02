#ifndef RUNGE_KUTTA_HH
#define RUNGE_KUTTA_HH
#include <functional>
#include <cmath>

#include <vector>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <optional>
#include <iostream>
#include <format>

/**
 * @file runge_kutta.hh
 * @author Simone Rodini (rodini.simone.luigi@gmail.com)
 * @brief  Default includes across Honeycomb
 * @version 0.1
 * @date 2025-08-02
 *
 * @copyright Copyright (c) 2025
 *
 */
namespace rk 
{
/**
 * @brief Butcher Tableau for a generic Runge-Kutta method
 *
 * @tparam Stages
 */
template <size_t Stages>
struct ButcherTableau {

      /// Delete default constructor to avoid non-sense
      ButcherTableau() = delete;

      ButcherTableau(double order,
                     const std::array<double, Stages>  &ci, 
                     const std::array<double, Stages>  &bi,
                     const std::array<std::vector<double>, Stages>  &ai)
          : _order(order), _aij(ai), _bi(bi), _ci(ci)
      {
         double s = 0.;
         /// Sanity checks
         for (size_t i = 0; i < Stages; i++) {
            s += bi[i];
            if (ai[i].size() > i)
               throw std::invalid_argument(
                   "Implicit methods (for which aij != 0  if j>=i) are not supported.");
         }
         if (std::fabs(s - 1.) > 1.0e-14) {
            std::printf("%.16e\n", s);
            throw std::invalid_argument("ButcherTableau: Sum of bi must be 1;");
         }
      };

      static constexpr size_t stages = Stages;
      const double _order;
      const std::array<std::vector<double>, Stages> _aij;
      const std::array<double, Stages> _bi;
      const std::array<double, Stages> _ci;
};

template <size_t Stages>
struct ButcherTableauWErrorEstimate : public ButcherTableau<Stages> {

      /// Delete default constructor to avoid non-sense
      ButcherTableauWErrorEstimate() = delete;

      ButcherTableauWErrorEstimate(double hi, double lo,
                                   const std::array<double, Stages>  &ci,
                                   const std::array<double, Stages>  &bi,
                                   const std::array<double, Stages>  &bi_lo,
                                   const std::array<std::vector<double>, Stages>  &ai)
          : ButcherTableau<Stages>(hi, ci, bi, ai), _order_lo(lo), _bi_lo(bi_lo)
      {
         // Note: depending on the method, not all bi_lo will sum to 1.
      };

      const double _order_lo;
      const std::array<double, Stages> _bi_lo;
};

/// Minimal required interface
template <typename S>
concept RungeKuttaCompatible = requires {
   // Exact match for: void clone(const S& other)
   { static_cast<void (S::*)(const S &)>(&S::clone) };

   // Exact match for: void add_with_weight(double, const S&)
   // Must implement S += w * S1, i.e. S = S + w * S1
   { static_cast<void (S::*)(double, const S &)>(&S::add_with_weight) };

   // Multiply by scalar
   { static_cast<void (S::*)(double)>(&S::scalar_mult) };
};

struct TimeInfo {
      TimeInfo(const std::vector<double> &ts = {0., 1.}, double dt_min = 0.1,
               size_t n_step_min = 10)
      {
         if (ts.size() <= 1) {
            throw std::invalid_argument(
                "TimeInfo: at least 2 time points are required (initial and final)");
         }
         _ts = ts;
         std::sort(_ts.begin(), _ts.end());

         double nd = static_cast<double>(n_step_min);
         for (size_t i = 0; i < _ts.size() - 1; i++) {
            size_t n_step = n_step_min;

            double dt_tmp = (_ts[i + 1] - _ts[i]) / nd;
            while (dt_tmp > dt_min) {
               n_step *= 2;
               dt_tmp = (_ts[i + 1] - _ts[i]) / static_cast<double>(n_step);
            }
            _dt.push_back(dt_tmp);
            _n_step.push_back(n_step);
         }
      }

      std::vector<double> _ts, _dt;
      std::vector<size_t> _n_step;
};

/// Useful aliases
template <typename Solution>
using rk_rhs_t = std::function<void(double, Solution &)>;

template <typename Solution>
using rk_callback_t = std::function<void(double, const TimeInfo &, const Solution &)>;

template <typename Solution, size_t Stages>
requires RungeKuttaCompatible<Solution>
struct RungeKutta {
   public:
      RungeKutta(const ButcherTableau<Stages> &tableau, const Solution &initial_conditions,
                 TimeInfo time_info, rk_rhs_t<Solution> &rhs)
          : _cb_each_step(false), _tableau(tableau), _solution(initial_conditions),
            _temp_step(initial_conditions), _time_info(time_info), _rhs(rhs)
      {
         for (size_t i = 0; i < Stages; i++) {
            _ki.push_back(initial_conditions);
         }
      }

      void AddCallback(rk_callback_t<Solution> &callback)
      {
         _callback = callback;
      }


      void CallbackEachStep()
      {
         _cb_each_step = true;
      }

      void CallbackOnlyOnTimeStamp()
      {
         _cb_each_step = false;
      }

      void step()
      {

         _temp_step.clone(_solution);

         for (size_t i = 0; i < Stages; i++) {

            _ki[i].clone(_solution);

            for (size_t j = 0; j < i; j++) {
               if (_tableau._aij[i][j] == 0.) continue;
               _ki[i].add_with_weight(_tableau._aij[i][j], _ki[j]);
            }

            _rhs(_t + _dt * _tableau._ci[i], _ki[i]);
            _ki[i].scalar_mult(_dt);

            _temp_step.add_with_weight(_tableau._bi[i], _ki[i]);
         }

         _solution.clone(_temp_step);
      }

      /// @name For visualization applications
      /// @{
      void advance_t() { _t += _dt; }
      void set_dt(double dt) { _dt = dt; }
      /// @}

      void operator()()
      {
         for (size_t i = 0; i < _time_info._ts.size() - 1; i++) {
            _t  = _time_info._ts[i];
            set_dt(_time_info._dt[i]);
            for (size_t j = 0; j < _time_info._n_step[i]; j++) {
               step();
               advance_t();
               if (_callback && _cb_each_step) {
                  (*_callback)(_t, _time_info, _solution);
               }
            }

            if (_callback && !_cb_each_step) {
               (*_callback)(_t, _time_info, _solution);
            }

            if (std::fabs(_t - _time_info._ts[i + 1]) > 1.0e-12) {
               std::cerr << std::fabs(_t - _time_info._ts[i + 1]) << std::endl;
               throw std::runtime_error("RungeKutta: time steps do not match. This is a bug.");
            }
         }
      }

      const Solution &GetSolution()
      {
         return _solution;
      }

   private:
      double _t, _dt;

      bool _cb_each_step;
      ButcherTableau<Stages> _tableau;
      Solution _solution;
      Solution _temp_step;
      std::vector<Solution> _ki;

      TimeInfo _time_info;

      /// The right hand side of ODE
      std::reference_wrapper<rk_rhs_t<Solution>> _rhs;
      /// The (optional) callback
      std::optional<std::reference_wrapper<rk_callback_t<Solution>>> _callback;
};

template <typename S>
concept AdaptiveRungeKuttaCompatible = RungeKuttaCompatible<S> && requires(const S &s) {
   { s.norm() } -> std::convertible_to<double>; 
   { s.make_zero() } -> std::same_as<void>; 
};

template <typename Solution, size_t Stages>
requires RungeKuttaCompatible<Solution>
struct AdaptiveRungeKutta {
   public:
      AdaptiveRungeKutta(const ButcherTableauWErrorEstimate<Stages> &tableau, const Solution &initial_conditions,
                 TimeInfo time_info, rk_rhs_t<Solution> &rhs, double toll)
          : _cb_each_step(false), _tableau(tableau), _solution(initial_conditions),
            _temp_step(initial_conditions), _err_step(initial_conditions),
            _time_info(time_info), _rhs(rhs)
      {
         for (size_t i = 0; i < Stages; i++) {
            _ki.push_back(initial_conditions);
         }
         if(toll < 1.0e-15){
            throw std::invalid_argument("AdaptiveRungeKutta: tollerance too small");
         }
         _tollerance = toll;
         _err_est = 0.;
         _t = 0.;
         _dt = 0.;
      }

      void AddCallback(rk_callback_t<Solution> &callback)
      {
         _callback = callback;
      }


      void CallbackEachStep()
      {
         _cb_each_step = true;
      }

      void CallbackOnlyOnTimeStamp()
      {
         _cb_each_step = false;
      }

      void try_step()
      {

         _temp_step.clone(_solution);
         _err_step.make_zero();

         for (size_t i = 0; i < Stages; i++) {
            _ki[i].clone(_solution);
            for (size_t j = 0; j < i; j++) {
               if (_tableau._aij[i][j] == 0.) continue;
               _ki[i].add_with_weight(_tableau._aij[i][j], _ki[j]);
            }
            _rhs(_t + _dt * _tableau._ci[i], _ki[i]);
            _ki[i].scalar_mult(_dt);
            _temp_step.add_with_weight(_tableau._bi[i], _ki[i]);
            _err_step.add_with_weight(_tableau._bi[i]-_tableau._bi_lo[i], _ki[i]);
         }
         _err_est = _err_step.norm();
      }

      void operator()()
      {
         for (size_t i = 0; i < _time_info._ts.size() - 1; i++) {
            _t  = _time_info._ts[i];
            _dt = _time_info._dt[i];
            while(_t - _time_info._ts[i + 1] < -1.0e-12) {
               try_step();
               double _dt_new = 0.9 * _dt * pow(_tollerance / _err_est, 1. / _tableau._order);

               // Failed step, try again with new _dt;
               if (_err_est > _tollerance) {
                  if (_dt_new < 1.0e-15 || _dt_new < 1.0e-6 * _time_info._dt[i]) {
                     std::fprintf(stderr, "%.10e\n", _dt_new);
                     throw std::runtime_error(
                         "Time step became too small, tollerance cannot be reached.");
                  }
                  _dt = _dt_new;
                  continue;
               }

               
               _t += _dt;
               _solution.clone(_temp_step);
               _dt = _dt_new;
               if(_t + _dt > _time_info._ts[i + 1]) _dt = (_time_info._ts[i + 1] - _t);
               if (_callback && _cb_each_step) {
                  (*_callback)(_t, _time_info, _solution);
               }
            }

            if (_callback && !_cb_each_step) {
               (*_callback)(_t, _time_info, _solution);
            }

            if (std::fabs(_t - _time_info._ts[i + 1]) > 1.0e-12) {
               std::cerr << std::fabs(_t - _time_info._ts[i + 1]) << std::endl;
               throw std::runtime_error("RungeKutta: time steps do not match. This is a bug.");
            }
         }
      }

      const Solution &GetSolution()
      {
         return _solution;
      }

   private:
      double _t, _dt;
      double _err_est, _tollerance;

      bool _cb_each_step;
      ButcherTableauWErrorEstimate<Stages> _tableau;
      Solution _solution;
      Solution _temp_step;
      Solution _err_step;
      std::vector<Solution> _ki;

      TimeInfo _time_info;

      /// The right hand side of ODE
      std::reference_wrapper<rk_rhs_t<Solution>> _rhs;
      /// The (optional) callback
      std::optional<std::reference_wrapper<rk_callback_t<Solution>>> _callback;
};

template <size_t N>
class vd
{

   public:
      vd()
      {
         _data.fill(0.);
      }
      vd(const std::array<double, N> &in)
      {
         for (size_t i = 0; i < N; i++) {
            _data[i] = in[i];
         }
      }

      double &operator[](size_t i)
      {
         if (i < N) return _data[i];
         else throw std::out_of_range(std::format("vs[{:d}], index out of range [0,{:d})", i, N));
      }
      const double &operator()(size_t i) const
      {
         if (i < N) return _data[i];
         else throw std::out_of_range(std::format("vs[{:d}], index out of range [0,{:d})", i, N));
      }

      inline size_t size() const
      {
         return N;
      }

      void clone(const vd<N> &other)
      {
         for (size_t i = 0; i < N; i++) {
            _data[i] = other(i);
         }
      }

      void add_with_weight(double w, const vd<N> &other)
      {
         for (size_t i = 0; i < N; i++) {
            _data[i] += w * other(i);
         }
      }

      void scalar_mult(double s)
      {
         for (size_t i = 0; i < N; i++) {
            _data[i] *= s;
         }
      }

      double distance2(const vd<N> &other)
      {
         double s = 0;
         for(size_t i=0; i< N; i++)
         {
            s += (_data[i] - other(i)) * (_data[i] - other(i));
         }
         return s;
      }

      double norm2()
      {
         double s = 0;
         for(size_t i=0; i< N; i++)
         {
            s += _data[i] * _data[i];
         }
         return s;
      }

   private:
      std::array<double, N> _data;
};


template <size_t Stages>
struct ButcherNystromTableauWErrorEstimate : public ButcherTableauWErrorEstimate<Stages> {

      /// Delete default constructor to avoid non-sense
      ButcherNystromTableauWErrorEstimate() = delete;

      ButcherNystromTableauWErrorEstimate(double hi, double lo,
                            const std::array<double, Stages>  &ci,
                            const std::array<double, Stages>  &bi,
                            const std::array<double, Stages>  &bi_lo,
                            const std::array<double, Stages>  &bbari,
                            const std::array<double, Stages>  &bbari_lo,
                            const std::array<std::vector<double>, Stages>  &ai)
          : ButcherTableauWErrorEstimate<Stages>(hi, lo, ci, bi, bi_lo, ai), _bbari(bbari),
            _bbari_lo(bbari_lo) {};

      const std::array<double, Stages> _bbari;
      const std::array<double, Stages> _bbari_lo;
};


template <size_t Stages>
struct ButcherNystromTableau : public ButcherTableau<Stages> {

      /// Delete default constructor to avoid non-sense
      ButcherNystromTableau() = delete;

      ButcherNystromTableau(double hi, 
                            const std::array<double, Stages>  &ci,
                            const std::array<double, Stages>  &bi,
                            const std::array<double, Stages>  &bbari,
                            const std::array<std::vector<double>, Stages>  &ai)
          : ButcherTableau<Stages>(hi, ci, bi, ai), _bbari(bbari) {};


      ButcherNystromTableau(const ButcherNystromTableauWErrorEstimate<Stages> &in_tab)
         : ButcherNystromTableau(in_tab._order, in_tab._ci, in_tab._bi, in_tab._bbari, in_tab._aij)
      {};   

      const std::array<double, Stages> _bbari;
};


template <size_t N>
using rkn_rhs_t = std::function<void(double, vd<N> &)>;

template <size_t N>
using rkn_callback_t = std::function<void(double, const TimeInfo &, const vd<N> &, const vd<N> &)>;


/**
 * @brief Class for solving Runge-Kutta-Nystrom ODE
 * 
 * @tparam Stages The order of the method to be used
 * @tparam N     The size of the vector of solution
 *
 * @note Only vector-like solutions are accepted, and they must be packaged 
 * into a vd<N> class.
 *
 * This class solves ODEs of the form \f$  f''(t) = RHS(t, f(t)) \f$, 
 * i.e. the right-hand side cannot depend on the first derivatives of f.
 * Problem of this form are the typical problems in Lagrangian Mechanic with
 * velocity-independent potentials. If the rhs of the ODE depends over 
 * the first derivative, use RungeKutta class. 
 * More details here: https://doi.org/10.1016/j.cam.2021.113753
 */
template <size_t Stages, size_t N>
struct RungeKuttaNystrom {
   public:
      RungeKuttaNystrom(const ButcherNystromTableau<Stages> &tableau,
                        const vd<N> &initial_conditions, const vd<N> &initial_conditions_der,
                        TimeInfo time_info, rkn_rhs_t<N> &rhs, double toll = 1.0e-12)
          : _tollerance(toll), _cb_each_step(false), _tableau(tableau), _solution(initial_conditions),
            _solution_der(initial_conditions_der), _temp_step(initial_conditions),
            _temp_step_der(initial_conditions_der), _err_step(initial_conditions),
            _err_step_der(initial_conditions_der), _time_info(time_info), _rhs(rhs)
      {
         for (size_t i = 0; i < Stages; i++) {
            _ki.push_back(initial_conditions);
         }
      }
      RungeKuttaNystrom(const ButcherNystromTableauWErrorEstimate<Stages> &tableau,
                        const vd<N> &initial_conditions, const vd<N> &initial_conditions_der,
                        TimeInfo time_info, rkn_rhs_t<N> &rhs, double toll = 1.0e-12)
          : _tollerance(toll), _cb_each_step(false), _tableau(tableau), _solution(initial_conditions),
            _solution_der(initial_conditions_der), _temp_step(initial_conditions),
            _temp_step_der(initial_conditions_der), _err_step(initial_conditions),
            _err_step_der(initial_conditions_der), _time_info(time_info), _rhs(rhs)
      {
         for (size_t i = 0; i < Stages; i++) {
            _ki.push_back(initial_conditions);
         }
      }

      void AddCallback(rkn_callback_t<N> &callback)
      {
         _callback = callback;
      }

      void CallbackEachStep()
      {
         _cb_each_step = true;
      }

      void CallbackOnlyOnTimeStamp()
      {
         _cb_each_step = false;
      }


      /// Note: no error estimation
      void step()
      {

         _temp_step.clone(_solution);
         _temp_step_der.clone(_solution_der);
         _temp_step.add_with_weight(_dt, _solution_der);

         for (size_t i = 0; i < Stages; i++) {

            _ki[i].clone(_solution);
            _ki[i].add_with_weight(_tableau._ci[i] * _dt, _solution_der);

            for (size_t j = 0; j < i; j++) {
               if (_tableau._aij[i][j] == 0.) continue;
               _ki[i].add_with_weight(_tableau._aij[i][j] * _dt, _ki[j]);
            }

            _rhs(_t + _tableau._ci[i] * _dt, _ki[i]);
            _ki[i].scalar_mult(_dt);

            _temp_step_der.add_with_weight(_tableau._bi[i], _ki[i]);
            _temp_step.add_with_weight(_tableau._bbari[i] * _dt, _ki[i]);
         }

         _solution.clone(_temp_step);
         _solution_der.clone(_temp_step_der);
      }

      /// @name For visualization applications
      /// @{
      void advance_t() { _t += _dt; }
      void set_dt(double dt) { _dt = dt; }
      /// @}

      void operator()()
      {
         for (size_t i = 0; i < _time_info._ts.size() - 1; i++) {
            _t  = _time_info._ts[i];
            set_dt(_time_info._dt[i]);
            for (size_t j = 0; j < _time_info._n_step[i]; j++) {
               step();
               advance_t();
               if (_callback && _cb_each_step) {
                  (*_callback)(_t, _time_info, _solution, _solution_der);
               }
            }

            if (_callback && !_cb_each_step) {
               (*_callback)(_t, _time_info, _solution, _solution_der);
            }

            if (std::fabs(_t - _time_info._ts[i + 1]) > 1.0e-12) {
               std::cerr << std::fabs(_t - _time_info._ts[i + 1]) << std::endl;
               throw std::runtime_error("RungeKutta: time steps do not match. This is a bug.");
            }
         }
      }

      std::pair<const vd<N> &, const vd<N> &> GetSolution()
      {
         return {_solution, _solution_der};
      }

   private:
      double _tollerance;
      double _t, _dt;
      double _err_est;

      bool _cb_each_step;
      ButcherNystromTableau<Stages> _tableau;
      vd<N> _solution, _solution_der;
      vd<N> _temp_step, _temp_step_der;
      vd<N> _err_step, _err_step_der;
      std::vector<vd<N>> _ki;

      TimeInfo _time_info;

      /// The right hand side of ODE
      std::reference_wrapper<rkn_rhs_t<N>> _rhs;
      /// The (optional) callback
      std::optional<std::reference_wrapper<rkn_callback_t<N>>> _callback;
};

/**
 * @brief Class for solving Runge-Kutta-Nystrom ODE
 * 
 * @tparam Stages The order of the method to be used
 * @tparam N     The size of the vector of solution
 *
 * @note Only vector-like solutions are accepted, and they must be packaged 
 * into a vd<N> class.
 *
 * This class solves ODEs of the form \f$  f''(t) = RHS(t, f(t)) \f$, 
 * i.e. the right-hand side cannot depend on the first derivatives of f.
 * Problem of this form are the typical problems in Lagrangian Mechanic with
 * velocity-independent potentials. If the rhs of the ODE depends over 
 * the first derivative, use RungeKutta class. 
 * More details here: https://doi.org/10.1016/j.cam.2021.113753
 */
template <size_t Stages, size_t N>
struct AdaptiveRungeKuttaNystrom {
   public:
      AdaptiveRungeKuttaNystrom(const ButcherNystromTableauWErrorEstimate<Stages> &tableau,
                        const vd<N> &initial_conditions, const vd<N> &initial_conditions_der,
                        TimeInfo time_info, rkn_rhs_t<N> &rhs, double toll = 1.0e-12)
          : _tollerance(toll), _cb_each_step(false), _tableau(tableau), _solution(initial_conditions),
            _solution_der(initial_conditions_der), _temp_step(initial_conditions),
            _temp_step_der(initial_conditions_der), _err_step(initial_conditions),
            _err_step_der(initial_conditions_der), _time_info(time_info), _rhs(rhs)
      {
         for (size_t i = 0; i < Stages; i++) {
            _ki.push_back(initial_conditions);
         }
      }

      void AddCallback(rkn_callback_t<N> &callback)
      {
         _callback = callback;
      }

      void CallbackEachStep()
      {
         _cb_each_step = true;
      }

      void CallbackOnlyOnTimeStamp()
      {
         _cb_each_step = false;
      }

      void try_step()
      {

         _temp_step.clone(_solution);
         _temp_step_der.clone(_solution_der);
         _temp_step.add_with_weight(_dt, _solution_der);

         for (size_t i = 0; i < Stages; i++) {

            _ki[i].clone(_solution);
            _ki[i].add_with_weight(_tableau._ci[i] * _dt, _solution_der);

            for (size_t j = 0; j < i; j++) {
               if (_tableau._aij[i][j] == 0.) continue;
               _ki[i].add_with_weight(_tableau._aij[i][j] * _dt, _ki[j]);
            }

            _rhs(_t + _tableau._ci[i] * _dt, _ki[i]);
            _ki[i].scalar_mult(_dt);

            _temp_step_der.add_with_weight(_tableau._bi[i], _ki[i]);
            _temp_step.add_with_weight(_tableau._bbari[i] * _dt, _ki[i]);
         }
         _err_step.clone(_ki[0]);
         _err_step.scalar_mult((_tableau._bbari[0] - _tableau._bbari_lo[0]) * _dt);
         _err_step_der.clone(_ki[0]);
         _err_step_der.scalar_mult(_tableau._bi[0] - _tableau._bi_lo[0]);

         for(size_t i=1; i< Stages; i++)
         {
            _err_step.add_with_weight((_tableau._bbari[i] - _tableau._bbari_lo[i]) * _dt,_ki[i]);
            _err_step_der.add_with_weight(_tableau._bi[i] - _tableau._bi_lo[i],_ki[i]);
         }
         _err_est = sqrt(_err_step.norm2() + _err_step_der.norm2());
         
      }


      void operator()()
      {
         for (size_t i = 0; i < _time_info._ts.size() - 1; i++) {
            _t  = _time_info._ts[i];
            _dt = _time_info._dt[i];

            while(_t - _time_info._ts[i + 1] < -1.0e-12) {
               try_step();
               double _dt_new = 0.9 * _dt * pow(_tollerance / _err_est, 1. / _tableau._order);

               // Failed step, try again with new _dt;
               if (_err_est > _tollerance) {
                  if (_dt_new < 1.0e-15 || _dt_new < 1.0e-6 * _time_info._dt[i]) {
                     std::fprintf(stderr, "%.10e\n", _dt_new);
                     throw std::runtime_error(
                         "Time step became too small, tollerance cannot be reached.");
                  }
                  _dt = _dt_new;
                  continue;
               }

               
               _t += _dt;
               _solution.clone(_temp_step);
               _solution_der.clone(_temp_step_der);
               _dt = _dt_new;
               if(_t + _dt > _time_info._ts[i + 1]) _dt = (_time_info._ts[i + 1] - _t);


               if (_callback && _cb_each_step) {
                  (*_callback)(_t, _time_info, _solution, _solution_der);
               }
            }

            if (_callback && !_cb_each_step) {
               (*_callback)(_t, _time_info, _solution, _solution_der);
            }

            if (std::fabs(_t - _time_info._ts[i + 1]) > 1.0e-12) {
               std::cerr << _t - _time_info._ts[i + 1] << std::endl;
               throw std::runtime_error("RungeKutta: time steps do not match. This is a bug.");
            }
         }
      }

      std::pair<const vd<N> &, const vd<N> &> GetSolution()
      {
         return {_solution, _solution_der};
      }

   private:
      double _tollerance;
      double _t, _dt;
      double _err_est;

      bool _cb_each_step;
      ButcherNystromTableauWErrorEstimate<Stages> _tableau;
      vd<N> _solution, _solution_der;
      vd<N> _temp_step, _temp_step_der;
      vd<N> _err_step, _err_step_der;
      std::vector<vd<N>> _ki;

      TimeInfo _time_info;

      /// The right hand side of ODE
      std::reference_wrapper<rkn_rhs_t<N>> _rhs;
      /// The (optional) callback
      std::optional<std::reference_wrapper<rkn_callback_t<N>>> _callback;
};

struct PreImplementedTableau {
   static inline const ButcherTableau<4> RKOriginal = 
   ButcherTableau<4>(4,
      {0,         0.5,       0.5,       1},         /* ci  */
      {1. / 6., 1. / 3., 1. / 3., 1. / 6.},         /* bi  */
      {{                                            /* aij */
         { }, 
         {0.5}, 
         {0, 0.5}, 
         {0, 0, 1}
      }}
   );

   // static inline const ButcherNystromTableau<3> RKNv1 = 
   // ButcherNystromTableau<3>(
   //    {(3. + sqrt(3.)) / 6.,  (3. - sqrt(3.)) / 6., (3. + sqrt(3.)) / 6.},        /* ci    */
   //    {(3. - 2. * sqrt(3.)) / 12., 0.5, (3. + 2. * sqrt(3.)) / 12.},              /* bi    */
   //    {(5. - 3. * sqrt(3.)) / 24., (3. + sqrt(3.)) / 12., (1. + sqrt(3.)) / 24.}, /* bbari */
   //    {{                                                                          /* aij   */
   //       { }, 
   //       {(2. - sqrt(3.)) / 12.}, 
   //       {0., sqrt(3.) / 6.}
   //    }}
   // );


   // static inline const ButcherNystromTableau<3> RKNv2 = 
   // ButcherNystromTableau<3>(
   //    {(3. - sqrt(3.)) / 6.,  (3. + sqrt(3.)) / 6., (3. - sqrt(3.)) / 6.},        /* ci    */
   //    {(3. + 2. * sqrt(3.)) / 12., 0.5, (3. - 2. * sqrt(3.)) / 12.},              /* bi    */
   //    {(5. + 3. * sqrt(3.)) / 24., (3. - sqrt(3.)) / 12., (1. - sqrt(3.)) / 24.}, /* bbari */
   //    {{                                                                          /* aij   */
   //       { }, 
   //       {(2. + sqrt(3.)) / 12.}, 
   //       {0., -sqrt(3.) / 6.}
   //    }}
   // );

   /// From Table 1 of doi.org/10.1016/j.cam.2021.113753
   static inline const ButcherNystromTableauWErrorEstimate<7> NEW7 = 
   ButcherNystromTableauWErrorEstimate<7>(7, 5,/* Order of the lowest step, for error esitmation */
      {0., 108816483. / 943181462, 108816483. / 471590731., 151401202. / 200292705., 682035803. / 631524599., 493263404. / 781610081., 1.},                            /* ci       */
      {53103334. / 780726093., 0., 244481296. / 685635505., 41493456. / 602487871., -45498718. / 926142189., 1625563237. / 4379140271., 191595797. / 1038702495.},     /* bi       */
      {41808761. / 935030896., 0., 224724272. / 506147085., 2995752066. / 3862177123., 170795979. / 811534085., -177906423. / 1116903503., -655510901. / 2077404990.}, /* bi_lo    */
      {53103334. / 780726093., 0., 352190060. / 1283966121., 37088117. / 2206150964., 7183323. / 1828127386., 187705681. / 1370684829., 0.},                           /* bbari    */
      {53103334. / 780726093., 0., 46261019. / 135447428., 289298425. / 1527932372., -52260067. / 3104571287.,  49872919. / 848719175., 0.},                           /* bbari_lo */
      {{   /* aij   */
         { }, 
         {5107771. / 767472028.}, 
         {5107771. / 575604021., 16661485. / 938806552.},
         {325996677. / 876867260., -397622579. / 499461366., 541212017. / 762248206.},
         {82243160. / 364375691., -515873404. / 1213273815., 820109726. / 1294837243., 36245507. / 242779260.},
         {3579594. / 351273191., 34292133. / 461028419., 267156948. / 2671391749., 22665163. / 1338599875., -3836509. / 1614789462.},
         {53103334. / 780726093., 0., 352190060. / 1283966121., 37088117. / 2206150964., 7183323. / 1828127386., 187705681. / 1370684829.},
      }}
   );

   /// Classical velocity-Verlet algorithm
   static inline const ButcherNystromTableau<2> VEL_VERLET =
   ButcherNystromTableau<2>(2,
      {0., 1.},    /* ci       */
      {0.5, 0.5},  /* bi       */
      {0.5, 0.},   /* bbari    */
      {{
         { }, 
         {0.5}
      }}
   );

   /// Okunbor Skeel, 6-oder 7-stages Method 1 (Table 2) https://doi.org/10.1016/0377-0427(92)00119-T
   static inline const ButcherNystromTableau<7> OS76 =
   ButcherNystromTableau<7>(6,
      {0.9441339218821262, 0.3462623051625522, 0.9347913701231966, 0.5, 0.06520862987680341024, 0.65373769483744778901, 0.05586607811787376572},    /* ci       */
      {-0.68774007118557290171, 0.13118241020105280626, 0.92161977504885189358, 0.26987577187133640373, 0.92161977504885189358, 0.13118241020105280626, -0.68774007118557290171},  /* bi       */
      {-0.03842134054164531, 0.08575888644805676, 0.06009756279830341, 0.1349378859356682, 0.8615222122505484919, 0.04542352375299604894, -0.6493187306439276215},   /* bbari    */
      {{
         { }, 
         {0.4111802682425534},
         {0.006425247211741155, 0.0772046612149093},
         {0.30544869505114114, 0.02016768134753036, -0.4007123247261225},
         {0.6044721428905411, -0.036869298519848596, -0.801424649452245, -0.11733965661499358},
         {0.1997171218597289, 0.04033536269506072, -0.2590246249935048, 0.0414900790599762, 0.5424000244587402155},
         {0.6108973901022823, -0.038094876977013074, -0.810034929902692, -0.11986098498218263, -0.0086102804504469946, -0.07843023967207378087}
      }}
   );

   /// From Blanes & Moan https://doi.org/10.1016/S0377-0427(01)00492-7
   static inline const ButcherNystromTableau<12> BM_SRKN11 =
   ButcherNystromTableau<12>(6,
      {0, 0.123229775946271, 0.41378357374582897, 0.28673436112041195, 0.04040260005833696, 0.39761147285426496, 0.602388527145735, 0.9595973999416629, 0.7132656388795879, 0.5862164262541709, 0.876770224053729,1.}, /* ci */
      {0.0414649985182624, 0.198128671918067, -0.0400061921041533, 0.0752539843015807, -0.0115113874206879, 0.2366699247869311, 0.2366699247869311, -0.0115113874206879, 0.0752539843015807, -0.0400061921041533, 0.198128671918067, 0.0414649985182624}, /* bi */
      {0.0414649985182624, 0.17371332006907136, -0.02345228696333458, 0.05367608119110145, -0.011046297438613276, 0.1425672474120913, 0.0941026773748398, -0.0004650899820746256, 0.02157790311047926, -0.016553905140818722, 0.02441535184899564, 0.}, /* bbari */
      {{
         { },
         {0.005109722477017935},
         {0.017157535272252118, 0.057567038078777},
         {0.011889439858992799, 0.03239494631306702, 0.005082755206973852},
         {0.0016752937515528902, -0.016410438357400515, 0.014937550961377614, -0.01853744647994612},
         {0.016486959132746226, 0.054362881207006435, 0.0006469841749956587, 0.008343944425823983, -0.00411196972486115},
         {0.02497803938551617, 0.09493508701306928, -0.007545365997511521, 0.02375423365479821, -0.006469237731677502, 0.04846457003725152},
         {0.039789704766709504, 0.1657084065774762, -0.021835932783893473, 0.050635624560568304, -0.010581207456538651, 0.13300516709508822, 0.0845405970578367},
         {0.0295755586592696, 0.11690302190700869, -0.011981137029489712, 0.03209817808062219, -0.007745587120932783, 0.07470584773189465, 0.026241277694643135, 0.002835620335605868},
         {0.024307463246010276, 0.09173093014129871, -0.00689838182251586, 0.022537218628180866, -0.006283074412908255, 0.04463712013559839, -0.0038274499016531295, 0.004298133043630395, -0.009560959452441321},
         {0.036355276041244465, 0.1492979682200757, -0.018522332873876293, 0.044402549566553456, -0.009627751745931184, 0.11340246560737698, 0.06493789557012546, 0.0009534557106074674, 0.012304371485931272, -0.011623951051360434},
         {0.0414649985182624, 0.17371332006907136, -0.02345228696333458, 0.05367608119110145, -0.011046297438613276, 0.1425672474120913, 0.0941026773748398, -0.0004650899820746256, 0.02157790311047926, -0.016553905140818722, 0.02441535184899564}
      }}
   );

   static inline const ButcherTableau<13> DOPRI8 = 
   ButcherTableau<13>(8, 
      {0., 1. / 18, 1. / 12, 1. / 8, 5. / 16, 3. / 8, 59. / 400, 93. / 200, 5490023248. / 9719169821, 13. / 20, 1201146811. / 1299019798, 1., 1.},
      {14005451. / 335480064, 0., 0., 0., 0., -59238493. / 1068277825, 181606767. / 758867731, 561292985. / 797845732, -1041891430. / 1371343529, 760417239. / 1151165299, 118820643. / 751138087, -528747749. / 2220607170, 1. / 4},
      {{
         {},
         {1./18},
         {1./48, 1./16},
         {1./32, 0., 3./32},
         {5./16, 0., -75./64, 75./64},
         {3./80, 0., 0., 3./16, 3./20},
         {29443841./614563906, 0., 0., 77736538./692538347, -28693883./1125000000, 23124283./1800000000},
         {16016141./946692911, 0., 0., 61564180./158732637, 22789713./633445777, 545815736./2771057229, -180193667./1043307555},
         {39632708./573591083, 0., 0., -433636366./683701615, -421739975./2616292301, 100302831./723423059, 790204164./839813087, 800635310./3783071287},
         {246121993./1340847787, 0., 0., -37695042795./15268766246, -309121744./1061227803, -12992083./490766935, 6005943493./2108947869, 393006217./1396673457, 123872331./1001029789},
         {-1028468189./846180014, 0., 0., 8478235783./508512852, 1311729495./1432422823, -10304129995./1701304382, -48777925059./3047939560, 15336726248./1032824649, -45442868181./3398467696, 3065993473./597172653},
         {185892177./718116043, 0., 0., -3185094517./667107341, -477755414./1098053517, -703635378./230739211, 5731566787./1027545527, 5232866602./850066563, -4093664535./808688257, 3962137247./1805957418, 65686358./487910083},
         {403863854./491063109, 0., 0., -5068492393./434740067, -411421997./543043805, 652783627./914296604, 11173962825./925320556, -13158990841./6184727034, 3936647629./1978049680, -160528059./685178525, 248638103./1413531060, 0.}
      }}
   );


   static inline const ButcherTableauWErrorEstimate<7> TSIT54 = 
   ButcherTableauWErrorEstimate<7>(5, 4,
      {0., 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0},
      {0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.},
      {0.001780011052226, 0.000816434459657, -0.007880878010262, 0.144711007173263, -0.582357165452555, 0.458082105929187, 1./66.},
      {{
         {},
         {0.161},
         {-0.008480655492356992, 0.3354806554923570},
         {2.8971530571054944, -6.359448489975075, 4.362295432869581},
         {5.32586482843926, -11.74888356406283, 7.495539342889836, -0.09249506636175525},
         {5.661747395595482, -12.92096931784711, 8.159367898576159, 0.07158497328140100, 0.02826905039406838},
         {0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774},
      }}
   );

   static inline const ButcherTableauWErrorEstimate<13> DOPRI87 = 
   ButcherTableauWErrorEstimate<13>(8, 7,
      {0., 1. / 18, 1. / 12, 1. / 8, 5. / 16, 3. / 8, 59. / 400, 93. / 200, 5490023248. / 9719169821, 13. / 20, 1201146811. / 1299019798, 1., 1.},
      {14005451. / 335480064, 0., 0., 0., 0., -59238493. / 1068277825, 181606767. / 758867731, 561292985. / 797845732, -1041891430. / 1371343529, 760417239. / 1151165299, 118820643. / 751138087, -528747749. / 2220607170, 1. / 4},
      {13451932. / 455176623., 0., 0., 0., 0., -808719846. /976000145., 1757004468. /5645159321., 656045339. /265891186., -3867574721. /1518517206., 465885868. /322736535., 53011238. /667516719., 2. /45., 0.},
      {{
         {},
         {1./18},
         {1./48, 1./16},
         {1./32, 0., 3./32},
         {5./16, 0., -75./64, 75./64},
         {3./80, 0., 0., 3./16, 3./20},
         {29443841./614563906, 0., 0., 77736538./692538347, -28693883./1125000000, 23124283./1800000000},
         {16016141./946692911, 0., 0., 61564180./158732637, 22789713./633445777, 545815736./2771057229, -180193667./1043307555},
         {39632708./573591083, 0., 0., -433636366./683701615, -421739975./2616292301, 100302831./723423059, 790204164./839813087, 800635310./3783071287},
         {246121993./1340847787, 0., 0., -37695042795./15268766246, -309121744./1061227803, -12992083./490766935, 6005943493./2108947869, 393006217./1396673457, 123872331./1001029789},
         {-1028468189./846180014, 0., 0., 8478235783./508512852, 1311729495./1432422823, -10304129995./1701304382, -48777925059./3047939560, 15336726248./1032824649, -45442868181./3398467696, 3065993473./597172653},
         {185892177./718116043, 0., 0., -3185094517./667107341, -477755414./1098053517, -703635378./230739211, 5731566787./1027545527, 5232866602./850066563, -4093664535./808688257, 3962137247./1805957418, 65686358./487910083},
         {403863854./491063109, 0., 0., -5068492393./434740067, -411421997./543043805, 652783627./914296604, 11173962825./925320556, -13158990841./6184727034, 3936647629./1978049680, -160528059./685178525, 248638103./1413531060, 0.}
      }}
   );

   /// Next three tableau from https://en.wikipedia.org/wiki/Runge-Kutta-Fehlberg_method
   static inline const ButcherTableauWErrorEstimate<6> FEHL_F2 = 
   ButcherTableauWErrorEstimate<6>(5, 4,
      {0., 0.25, 3./8., 12./13., 1., 0.5},
      {16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.},
      {25./216., 0., 1408./2565., 2197/4104., -1./5., 0.},
      {{
         {},
         {1./4.},
         {3./32., 9./32.},
         {1932./2197., -7200./2197., 7296./2197.},
         {439./216., -8., 3680./513., -845./4104.},
         {-8./27., 2., -3544./2565., 1859./4104., -11./40.},
      }}
   );

   static inline const ButcherTableauWErrorEstimate<6> FEHL_F1 = 
   ButcherTableauWErrorEstimate<6>(5, 4,
      {0., 2./9., 1./3., 3./4., 1., 5./6.},
      {47./450., 0., 12./25., 32./225., 1./30., 6./25.},
      {1./9., 0., 9./20., 16./45., 1./12., 0.},
      {{
         {},
         {2./9.},
         {1./12., 1./4.},
         {69./128., -243./128., 135./64.},
         {-17./12., 27./4., -27./5., 16./15.},
         {65/432., -5./16., 13./16., 4./27., 5./144.},
      }}
   );

   static inline const ButcherTableauWErrorEstimate<6> FEHL_F4 = 
   ButcherTableauWErrorEstimate<6>(5, 4,
      {0., 0.5, 0.5, 1., 2./3., 1./5.},
      {1./24., 0., 0., 5./48., 27./56., 125./336.},
      {1./6., 0., 2./3., 1./6., 0., 0.},
      {{
         {},
         {0.5},
         {0.25, 0.25},
         {0., -1., 2.},
         {7./27., 10./27., 0., 1./27.},
         {28./625., -1./5., 546./625., 54./625., -378/625.},
      }}
   );

   static inline const ButcherTableauWErrorEstimate<7> DOPRI54 = 
   ButcherTableauWErrorEstimate<7>(5, 4, 
      {0., 1./5, 3./10, 4./5, 8./9, 1., 1.},
      {35./384, 0., 500./1113, 125./192, -2187./6784, 11./84, 0.},
      {5179./57600., 0., 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.},
      {{
         {},
         {1./5},
         {3./40, 9./40},
         {44./45, -56./15, 32./9},
         {19372./6561, -25360./2187, 64448./6561, -212./729},
         {9017./3168, -355./33, 46732./5247, 49./176, -5103./18656},
         {35./384, 0., 500./1113, 125./192, -2187./6784, 11./84}
      }}
   );
};



} // namespace rk
#endif // RUNGE_KUTTA_HH