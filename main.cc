#include "runge_kutta.hh"
#include "timer.hh"
#include "ran2.h"
#include <raylib.h>

Ran2 global_ran2(-2);

static inline double sq(double x)
{
   return x * x;
}

static constexpr size_t N      = 15 * 15;
static constexpr size_t Nsqrt  = 15;
static constexpr double L      = 6.;
static constexpr double sigma2 = L / Nsqrt;

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

      Vec2d &operator+=(const Vec2d &other)
      {
         x += other.x;
         y += other.y;
         return *this;
      }
      Vec2d &operator-=(const Vec2d &other)
      {
         x -= other.x;
         y -= other.y;
         return *this;
      }

      Vec2d &operator*=(double s)
      {
         x *= s;
         y *= s;
         return *this;
      }
};

Vec2d operator*(const Vec2d &p, double w)
{
   return {p.x * w, p.y * w};
}

double remap(double x, double xmin, double xmax)
{
   return (xmax - xmin) * x + xmin;
}

struct State {
      size_t _N;
      std::vector<std::vector<Vec2d>> Fik;
      std::vector<Vec2d> pos, vel;

      State() : _N(N), Fik(N), pos(N), vel(N)
      {

         Vec2d cm_vel{0, 0};

         for (size_t i = 0; i < N; i++) {
            Fik[i]   = std::vector<Vec2d>(i, Vec2d{.x = 0, .y = 0});
            pos[i].x = L * (double)(i % Nsqrt) / Nsqrt;
            size_t j = i / Nsqrt;
            pos[i].y = L * (double)j / Nsqrt;

            vel[i].x = global_ran2.uniform(-0.01, 0.01);
            vel[i].y = global_ran2.uniform(-0.01, 0.01);

            cm_vel += vel[i];
         }

         for (size_t i = 0; i < _N; i++) {
            vel[i] -= cm_vel * (1. / (double)_N);
         }
      }

      void clone(const State &other)
      {
         _N  = other._N;
         pos = other.pos;
         vel = other.vel;
         Fik = other.Fik;
      }

      void add_with_weight(double w, const State &other)
      {
         for (size_t i = 0; i < _N; i++) {
            pos[i] += other.pos[i] * w;
            vel[i] += other.vel[i] * w;
         }
      }

      void scalar_mult(double s)
      {
         for (size_t i = 0; i < _N; i++) {
            pos[i] *= s;
            vel[i] *= s;
         }
      }

      void pbb()
      {
         for (size_t i = 0; i < _N; i++) {
            if (pos[i].x < 0.) {
               pos[i].x = L + pos[i].x;
            }
            if (pos[i].x > L) {
               pos[i].x = pos[i].x - L;
            }

            if (pos[i].y < 0.) {
               pos[i].y = L + pos[i].y;
            }
            if (pos[i].y > L) {
               pos[i].y = pos[i].y - L;
            }
         }
      }

      std::pair<double, double> energy() const
      {
         double en  = 0.;
         double kin = 0.;
         for (size_t i = 0; i < _N; i++) {
            kin += (sq(vel[i].x) + sq(vel[i].y)) * 0.5;
            for (size_t j = i + 1; j < _N; j++) {
               double r12_sq =
                   sq(min_image(pos[i].x - pos[j].x, L)) + sq(min_image(pos[i].y - pos[j].y, L));

               en += 4. * (pow(sigma2 / r12_sq, 6) - pow(sigma2 / r12_sq, 3));
            }
         }

         return {kin / ((double)N), en + kin};
      }
};

void compute_force(const Vec2d &p1, const Vec2d &p2, Vec2d &out)
{
   double r12_sq = sq(min_image(p1.x - p2.x, L)) + sq(min_image(p1.y - p2.y, L));
   double pref   = 24.0 * (2. * pow(sigma2 / r12_sq, 6) - pow(sigma2 / r12_sq, 3)) / r12_sq;

   out.x = min_image(p1.x - p2.x) * pref;
   out.y = min_image(p1.y - p2.y) * pref;
}

void md_rhs(double, State &state)
{
   for (size_t i = 0; i < state._N; i++) {
      for (size_t k = 0; k < i; k++) {
         compute_force(state.pos[i], state.pos[k], state.Fik[i][k]);
      }
   }
   for (size_t i = 0; i < state._N; i++) {
      state.pos[i] = state.vel[i];
      state.vel[i] = Vec2d{.x = 0, .y = 0};
      for (size_t k = 0; k < i; k++) {
         state.vel[i] += state.Fik[i][k];
      }
      for (size_t k = i + 1; k < state._N; k++) {
         state.vel[i] -= state.Fik[k][i];
      }
   }
}

void quit()
{
   CloseWindow();
   exit(1);
}

int main()
{

   const double Width  = 800;
   const double Height = 800;

   InitWindow((int)Width, (int)Height, "MD - RK");

   SetTargetFPS(120);
   size_t frame_count = 0;
   timer::mark begin, end;

   State state;
   auto tb = rk::PreImplementedTableau::DOPRI8;

   rk::rk_rhs_t<State> _rhs = [&](double t, State &st) {
      return md_rhs(t, st);
   };

   auto solver = rk::RungeKutta<State, tb.stages>(tb, state, {}, _rhs);
   solver.set_dt(0.001);

   while (!WindowShouldClose()) {
      begin = timer::now();

      BeginDrawing();

      ClearBackground(RAYWHITE);

      solver.step();
      solver.advance_t();

      State &curr_state = const_cast<State &>(solver.GetSolution());
      // curr_state.bounce();
      const auto [kin, en] = curr_state.energy();
      curr_state.pbb();

      for (size_t i = 0; i < state._N; i++) {
         DrawCircle(curr_state.pos[i].x * Width / L, curr_state.pos[i].y * Height / L, 3, RED);
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