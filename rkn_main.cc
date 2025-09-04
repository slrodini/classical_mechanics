#include "runge_kutta.hh"
#include "timer.hh"
#include "ran2.h"
#include <raylib.h>

Ran2 global_ran2(-2);
static inline double sq(double x)
{
   return x * x;
}

static constexpr size_t Nsqrt = 15;
static constexpr size_t N     = Nsqrt * Nsqrt;
static constexpr double L     = 6.;
// static constexpr double sigma2 = L / Nsqrt; /* Get Lattice config */
static constexpr double sigma2 = L / Nsqrt * 0.8;

/// Works for d \in [0,1]
static inline double min_image(double d)
{
   return d - std::round(d);
}

/// Works for d \in [0,L]
static inline double min_image(double d, double l)
{
   return d - l * std::round(d / l);
}
struct Vec2d {
      double x, y;
};

void compute_force(const Vec2d &p1, const Vec2d &p2, Vec2d &out)
{
   double r12_sq = sq(min_image(p1.x - p2.x, L)) + sq(min_image(p1.y - p2.y, L));
   double pref   = 24.0 * (2. * pow(sigma2 / r12_sq, 6) - pow(sigma2 / r12_sq, 3)) / r12_sq;

   out.x = min_image(p1.x - p2.x, L) * pref;
   out.y = min_image(p1.y - p2.y, L) * pref;
}

std::vector<std::vector<Vec2d>> Fik;

void md_rhs(double, rk::vd<2 * N> &state)
{
   for (size_t i = 0; i < N; i++) {
      for (size_t k = 0; k < i; k++) {
         compute_force({state[2 * i], state[2 * i + 1]}, {state[2 * k], state[2 * k + 1]},
                       Fik[i][k]);
      }
   }
   for (size_t i = 0; i < N; i++) {
      state[2 * i]     = 0.;
      state[2 * i + 1] = 0.;
      for (size_t k = 0; k < i; k++) {
         state[2 * i] += Fik[i][k].x;
         state[2 * i + 1] += Fik[i][k].y;
      }
      for (size_t k = i + 1; k < N; k++) {
         state[2 * i] -= Fik[k][i].x;
         state[2 * i + 1] -= Fik[k][i].y;
      }
   }
}

void quit()
{
   CloseWindow();
   exit(1);
}

double remap(double x, double xmin, double xmax)
{
   return (xmax - xmin) * x + xmin;
}

void pbb(rk::vd<2 * N> &pos)
{
   for (size_t i = 0; i < N; i++) {
      if (pos[2 * i] < 0.) {
         pos[2 * i] = L + pos[2 * i];
      }
      if (pos[2 * i] > L) {
         pos[2 * i] = pos[2 * i] - L;
      }

      if (pos[2 * i + 1] < 0.) {
         pos[2 * i + 1] = L + pos[2 * i + 1];
      }
      if (pos[2 * i + 1] > L) {
         pos[2 * i + 1] = pos[2 * i + 1] - L;
      }
   }
}

std::pair<double, double> energy(rk::vd<2 * N> &pos, rk::vd<2 * N> &vel)
{
   double en  = 0.;
   double kin = 0.;
   for (size_t i = 0; i < N; i++) {

      kin += (sq(vel[2 * i]) + sq(vel[2 * i + 1])) * 0.5;

      for (size_t j = i + 1; j < N; j++) {
         double r12_sq = sq(min_image(pos[2 * i] - pos[2 * j], L)) +
                         sq(min_image(pos[2 * i + 1] - pos[2 * j + 1], L));

         en += 4. * (pow(sigma2 / r12_sq, 6) - pow(sigma2 / r12_sq, 3));
      }
   }

   return {kin / ((double)N), en + kin};
}

int main()
{
   Fik.resize(N);
   for (size_t i = 0; i < N; i++) {
      Fik[i] = std::vector<Vec2d>(i, Vec2d{.x = 0, .y = 0});
   }

   const double Width  = 800;
   const double Height = 800;

   InitWindow((int)Width, (int)Height, "MD-RKN");

   SetTargetFPS(120);
   size_t frame_count = 0;
   timer::mark begin, end;

   rk::vd<2 * N> pos, vel;
   double cm_vel_x = 0.;
   double cm_vel_y = 0.;

   for (size_t i = 0; i < N; i++) {
      pos[2 * i]     = L * (double)(i % Nsqrt) / Nsqrt;
      size_t j       = i / Nsqrt;
      pos[2 * i + 1] = L * (double)j / Nsqrt;

      vel[2 * i]     = global_ran2.uniform(-0.01, 0.01);
      vel[2 * i + 1] = global_ran2.uniform(-0.01, 0.01);

      cm_vel_x += vel[2 * i];
      cm_vel_y += vel[2 * i + 1];
   }

   for (size_t i = 0; i < N; i++) {
      vel[2 * i] -= cm_vel_x * (1. / (double)N);
      vel[2 * i + 1] -= cm_vel_y * (1. / (double)N);
   }

   // auto tb = rk::PreImplementedTableau::VEL_VERLET;
   // auto tb = rk::PreImplementedTableau::NEW7;
   auto tb = rk::PreImplementedTableau::BM_SRKN11;

   rk::rkn_rhs_t<2 * N> _rhs = [&](double t, rk::vd<2 * N> &st) {
      return md_rhs(t, st);
   };

   auto solver = rk::RungeKuttaNystrom<tb.stages, 2 * N>(tb, pos, vel, {}, _rhs);
   solver.set_dt(0.001);

   while (!WindowShouldClose()) {
      begin = timer::now();

      BeginDrawing();

      ClearBackground(RAYWHITE);

      solver.step();
      solver.advance_t();

      std::pair<const rk::vd<2 * N>, const rk::vd<2 * N>> curr_state = solver.GetSolution();

      rk::vd<2 * N> &curr_pos = const_cast<rk::vd<2 * N> &>(curr_state.first);
      rk::vd<2 * N> &curr_vel = const_cast<rk::vd<2 * N> &>(curr_state.second);

      const auto [kin, en] = energy(curr_pos, curr_vel);
      // bounce(curr_pos, curr_vel);
      pbb(curr_pos);

      for (size_t i = 0; i < N; i++) {
         DrawCircle(curr_pos[2 * i] * Width / L, curr_pos[2 * i + 1] * Height / L, 3, RED);
      }

      EndDrawing();
      end = timer::now();

      printf("Frame: %04zu", (frame_count++));
      printf(" took: %3.4f ms to render; FPS = %03zu", timer::elapsed_ms(end, begin),
             size_t(1. / timer::elapsed_s(end, begin)));
      printf(";  Energy = %+.4e", en);
      printf(";  T = %+.4e", kin);

      printf("\n");
   }

   CloseWindow();
}