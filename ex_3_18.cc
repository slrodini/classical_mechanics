#include "runge_kutta.hh"
#include "timer.hh"
#include <raylib.h>

/// Simulation parameters
// static constexpr double M     = 1.;
static constexpr double R     = 1.;
static constexpr double alpha = 0.; // M_PI / 4;
static constexpr double k_o_M = 1.;
static constexpr double g_o_R = 3.;

static constexpr double Width  = 800;
static constexpr double Height = 800;

struct Configuration {
      double xA, xB;
      double yA, yB;
      double xDisk;
};

struct State {
      double q, q_d, theta, theta_d;

      State(double q0, double q_d0, double t0, double t_d0)
          : q(q0), q_d(q_d0), theta(t0), theta_d(t_d0)
      {
      }

      void clone(const State &s)
      {
         q       = s.q;
         theta   = s.theta;
         q_d     = s.q_d;
         theta_d = s.theta_d;
      }

      void add_with_weight(double w, const State &s)
      {
         q += w * s.q;
         theta += w * s.theta;
         q_d += w * s.q_d;
         theta_d += w * s.theta_d;
      }

      void scalar_mult(double w)
      {
         q *= w;
         theta *= w;
         q_d *= w;
         theta_d *= w;
      }

      Configuration get_config() const
      {
         const double c = cos(theta);
         const double s = sin(theta);

         const double c_a = cos(alpha);
         const double s_a = sin(alpha);
         return Configuration{
             .xA    = R * (q + c_a),
             .xB    = R * (q + c_a + c),
             .yA    = R * (s_a),
             .yB    = R * (s_a + s),
             .xDisk = R * q,
         };
      }
};

void quit()
{
   CloseWindow();
   exit(1);
}

int main()
{

   // Space is mapped to [-4R,4R]^2
   const double scaling = Width / (8 * R);

   InitWindow((int)Width, (int)Height, "EX-3.18");
   SetTargetFPS(120);

   size_t frame_count = 0;
   timer::mark begin, end;

   auto tb = rk::PreImplementedTableau::DOPRI8;

   rk::rk_rhs_t<State> _rhs = [&](double, State &st) {
      // Cache variables
      double q = st.q, theta = st.theta, q_d = st.q_d, theta_d = st.theta_d;

      const double c = cos(theta);
      const double s = sin(theta);

      const double A = -2 * k_o_M * (q + cos(alpha) + c) + theta_d * theta_d * c;
      const double B = -g_o_R * c + 2 * k_o_M * (q * s + sin(theta - alpha)) + q_d * theta_d * c;

      const double den = 10 - 3 * s * s;

      st.q     = st.q_d;
      st.theta = st.theta_d;

      st.q_d     = (2 * A + 3 * B * s) / den;
      st.theta_d = (15 * B + 3 * A * s) / den;

      // printf("=====================================================\n");
      // printf("A: %.16e\t B: %.16e\n", A, B);
      // printf("=====================================================\n");

      if (st.theta < -M_PI) st.theta += 2 * M_PI;
      if (st.theta > M_PI) st.theta -= 2 * M_PI;
   };

   State state(-cos(alpha), 0, -M_PI / 2 + 0.1, 0);

   // _rhs(0, state);
   // quit();

   auto solver = rk::RungeKutta<State, tb.stages>(tb, state, {}, _rhs);
   solver.set_dt(0.001);

   while (!WindowShouldClose()) {
      begin = timer::now();

      BeginDrawing();

      ClearBackground(RAYWHITE);

      solver.step();
      solver.advance_t();

      auto curr_state = solver.GetSolution();
      auto conf       = curr_state.get_config();

      DrawLine(0, Height / 2, Width, Height / 2, LIGHTGRAY);
      DrawLine(Width / 2, 0, Width / 2, Height, LIGHTGRAY);
      DrawCircle((conf.xDisk + 4 * R) * scaling, Height / 2, R * scaling, RED);

      Vector2 in{(float)((conf.xA + 4 * R) * scaling),
                 (float)(Height - (conf.yA + 4 * R) * scaling)};
      Vector2 fin{(float)((conf.xB + 4 * R) * scaling),
                  (float)(Height - (conf.yB + 4 * R) * scaling)};

      Vector2 ancor_1{(float)((-2 * R + 4 * R) * scaling), (float)(Height / 2)};
      Vector2 ancor_2{(float)((+2 * R + 4 * R) * scaling), (float)(Height / 2)};

      DrawLineEx(in, fin, 7, BLUE);

      DrawLineEx(fin, ancor_1, 2, GREEN);
      DrawLineEx(fin, ancor_2, 2, GREEN);

      EndDrawing();
      end = timer::now();

      printf("Frame: %04zu", (frame_count++));
      printf(" took: %3.4f ms to render; FPS = %03zu", timer::elapsed_ms(end, begin),
             size_t(1. / timer::elapsed_s(end, begin)));

      printf(";  Theta: %3.4f\n", curr_state.theta);

      printf("\n");
   }

   CloseWindow();
}