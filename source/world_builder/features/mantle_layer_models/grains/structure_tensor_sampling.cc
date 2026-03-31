/*
  Copyright (C) 2018-2026 by the authors of the World Builder code.

  This file is part of the World Builder.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published
   by the Free Software Foundation, either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "world_builder/features/mantle_layer_models/grains/structure_tensor_sampling.h"


#include "world_builder/nan.h"
#include "world_builder/types/array.h"
#include "world_builder/types/double.h"
#include "world_builder/types/object.h"
#include "world_builder/types/one_of.h"
#include "world_builder/types/unsigned_int.h"
#include "world_builder/types/value_at_points.h"
#include "world_builder/utilities.h"
#include "world_builder/world.h"

#include <cmath>
#include <random>

namespace WorldBuilder
{
  using namespace Utilities;

  namespace Features
  {
    namespace MantleLayerModels
    {
      namespace Grains
      {

        StructureTensorSampling::StructureTensorSampling(WorldBuilder::World *world_)
          :
          min_depth(NaN::DSNAN),
          max_depth(NaN::DSNAN)

        {
          this->world = world_;
          this->name = "structure tensor sampling";
        }

        StructureTensorSampling::~StructureTensorSampling() = default;

        void
        StructureTensorSampling::declare_entries(Parameters &prm,
                                                 const std::string &)
        {
          // Document plugin and require entries if needed.
          // Add compositions the required parameters.
          prm.declare_entry("", Types::Object({"compositions"}),
                            "Uniform grains model. All grains start exactly the same.");

          // Declare entries of this plugin
          prm.declare_entry("min depth", Types::OneOf(Types::Double(0),Types::Array(Types::ValueAtPoints(0.,2))),
                            "The depth in meters from which the composition of this feature is present.");
          prm.declare_entry("max depth", Types::OneOf(Types::Double(std::numeric_limits<double>::max()),Types::Array(Types::ValueAtPoints(std::numeric_limits<double>::max(),2))),
                            "The depth in meters to which the composition of this feature is present.");

          prm.declare_entry("compositions", Types::Array(Types::UnsignedInt(),0),
                            "A list with the integer labels of the composition which are present there.");

          prm.declare_entry("structure tensors",
                            Types::Array(Types::Array(Types::Array(Types::Array(Types::Double(0),3),3),3),0),
                            "Three second order structure tensors (one for each axis). The tensors should be normalized so that their trace is equal to 1.");

          prm.declare_entry("n grains", Types::UnsignedInt(100),
                            "Number of grains to sample.");

          prm.declare_entry("tolerance", Types::Double(1e-4),
                            "Tolerance for misfit");

          prm.declare_entry("max iterations", Types::UnsignedInt(100000),
                            "Maximum number of Monte Carlo iterations");
        }

        void
        StructureTensorSampling::parse_entries(Parameters &prm, const std::vector<Point<2>> &coordinates)
        {
          min_depth_surface = Objects::Surface(prm.get("min depth",coordinates));
          min_depth = min_depth_surface.minimum;
          max_depth_surface = Objects::Surface(prm.get("max depth",coordinates));
          max_depth = max_depth_surface.maximum;
          compositions = prm.get_vector<unsigned int>("compositions");

          struct_tensors = prm.get_vector<std::array<std::array<std::array<double,3>,3>,3>>("structure tensors");

          n_grains = prm.get<unsigned int>("n grains");
          tolerance = prm.get<double>("tolerance");
          max_iterations = prm.get<unsigned int>("max iterations");

          WBAssertThrow(compositions.size() == struct_tensors.size(),
                        "There are not the same amount of compositions (" << compositions.size()
                        << ") and structure tensors (" << struct_tensors.size() << ").");
        }


        // Compute the second-order structure tensor from a set of axes
        std::array<std::array<double,3>,3>
        StructureTensorSampling::struct_tensor_from_axes(const std::vector<std::array<double,3>> &axes,
                                                         const std::vector<double> &weights) const
        {
          std::array<std::array<double,3>,3> a_tens = {{{{0.0}}}}; // initialize 3x3 tensor to zero

          size_t N = axes.size();
          std::vector<double> w;
          if (weights.empty())
            w = std::vector<double>(N, 1.0); // default weights = 1
          else
            w = weights;

          double wsum = std::accumulate(w.begin(), w.end(), 0.0);

          for (size_t i = 0; i < N; ++i)
            {
              const auto &n = axes[i];
              double wi = w[i];

              // Outer product n * n^T, scaled by weight
              for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 3; ++k)
                  a_tens[j][k] += n[j] * n[k] * wi;
            }

          // Normalize by sum of weights
          for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
              a_tens[j][k] /= wsum;

          return a_tens;
        }


        std::pair<double,double>
        StructureTensorSampling::compute_misfit(const std::array<std::vector<std::array<double,3>>,3> &axes_model,
                                                const std::array<std::array<std::array<double,3>,3>,3> &a2_data,
                                                const double sigma) const
        {
          // Number of grains
          // const size_t n_grains = axes_model[0].size();
          const double sigma_mean = sigma / std::sqrt(static_cast<double>(n_grains));

          double delta_a2 = 0.0;

          for (size_t i = 0; i < 3; ++i)
            {
              // Compute the modeled structure tensor from axes[i]
              std::array<std::array<double,3>,3> struct_tensor_model = struct_tensor_from_axes(axes_model[i]);

              // Sum squared deviations of upper triangle (including diagonal)
              delta_a2 += ( std::pow(a2_data[i][0][0] - struct_tensor_model[0][0],2) +
                            std::pow(a2_data[i][1][1] - struct_tensor_model[1][1],2) +
                            std::pow(a2_data[i][2][2] - struct_tensor_model[2][2],2) +
                            std::pow(a2_data[i][0][1] - struct_tensor_model[0][1],2) +
                            std::pow(a2_data[i][0][2] - struct_tensor_model[0][2],2) +
                            std::pow(a2_data[i][1][2] - struct_tensor_model[1][2],2) ) / (3.0*6.0);
            }

          double S = delta_a2 / (sigma_mean*sigma_mean);
          return {S, std::sqrt(delta_a2)};
        }


        WorldBuilder::grains
        StructureTensorSampling::get_grains(const Point<3> & /*position_in_cartesian_coordinates*/,
                                            const Objects::NaturalCoordinate &position_in_natural_coordinates,
                                            const double depth,
                                            const unsigned int composition_number,
                                            WorldBuilder::grains grains_,
                                            const double  /*feature_min_depth*/,
                                            const double /*feature_max_depth*/) const
        {
          WorldBuilder::grains  grains_local = grains_;
          if (depth <= max_depth && depth >= min_depth)
            {
              const double min_depth_local = min_depth_surface.constant_value ? min_depth : min_depth_surface.local_value(position_in_natural_coordinates.get_surface_point()).interpolated_value;
              const double max_depth_local = max_depth_surface.constant_value ? max_depth : max_depth_surface.local_value(position_in_natural_coordinates.get_surface_point()).interpolated_value;
              if (depth <= max_depth_local &&  depth >= min_depth_local)
                {
                  for (unsigned int i =0; i < compositions.size(); ++i)
                    {
                      if (compositions[i] == composition_number)
                        {
                          const unsigned long N = grains_local.rotation_matrices.size();
                          // std::cout << "N = " << N << std::endl;
                          // std::cout << "n_grains = " << n_grains << std::endl;
                          WBAssertThrow(N == n_grains, "The number of grains from input files should match.");

                          // --- 1. initialize uniform Euler angles ---
                          std::vector<std::array<double, 3>> euler(N);
                          std::vector<std::array<std::array<double, 3>, 3>> rotation_matrices(N);
                          std::array<std::vector<std::array<double,3>>, 3> axes_old; // 3 axes x N grains

                          std::uniform_real_distribution<> dist(0.0, 1.0);
                          auto &engine = world->get_random_number_engine();

                          for (unsigned int k = 0; k < N; ++k)
                            {
                              double phi1 = 2.0 * Consts::PI * dist(engine);
                              double theta = std::acos(1.0 - 2.0 * dist(engine));
                              double phi2 = 2.0 * Consts::PI * dist(engine);

                              euler[k] = {{phi1, theta, phi2}};
                              rotation_matrices[k] = WorldBuilder::Utilities::euler_angles_to_rotation_matrix(phi1, theta, phi2);
                            }

                          // --- 2. convert rotation matrices to axes vectors ---
                          for (unsigned int i_axis = 0; i_axis < 3; ++i_axis)
                            axes_old[i_axis].resize(N);
                          for (unsigned int g = 0; g < N; ++g)
                            {
                              const auto &R = rotation_matrices[g];
                              for (unsigned int i_axis = 0; i_axis < 3; ++i_axis)
                                {
                                  axes_old[i_axis][g] = {{ R[0][i_axis], R[1][i_axis], R[2][i_axis] }};
                                }
                            }

                          // --- 3. compute initial misfit ---
                          std::array<std::vector<std::array<double,3>>, 3> axes_new = axes_old;
                          double misfit_old, rms_diff_struct_tensor_old;
                          std::tie(misfit_old, rms_diff_struct_tensor_old) = compute_misfit(axes_old, struct_tensors[composition_number], tolerance);

                          std::vector<double> S_arr;
                          S_arr.reserve(max_iterations);

                          unsigned int n_acc = 0;
                          unsigned int n_iter = 0;

                          while (rms_diff_struct_tensor_old > tolerance)
                            {
                              S_arr.push_back(misfit_old);
                              ++n_iter;

                              if (n_iter > max_iterations)
                                {
                                  std::cout << "Warning: maximum number of iterations reached" << std::endl;
                                  break;
                                }

                              // pick a random grain to perturb
                              std::mt19937 engine_index {std::random_device{}()}; // your random engine
                              std::uniform_int_distribution<unsigned int> dist_index(0, n_grains - 1);
                              unsigned int random_index = dist_index(engine_index);
                              double phi1 = 2.0 * Consts::PI * dist(engine);
                              double theta = std::acos(1.0 - 2.0 * dist(engine));
                              double phi2 = 2.0 * Consts::PI * dist(engine);

                              std::array<double,3> random_EA = {{phi1, theta, phi2}};
                              std::array<std::array<double,3>,3> random_rotation_matrix = WorldBuilder::Utilities::euler_angles_to_rotation_matrix(phi1, theta, phi2);
                              axes_new[0][random_index] = {{random_rotation_matrix[0][0], random_rotation_matrix[1][0], random_rotation_matrix[2][0]}};
                              axes_new[1][random_index] = {{random_rotation_matrix[0][1], random_rotation_matrix[1][1], random_rotation_matrix[2][1]}};
                              axes_new[2][random_index] = {{random_rotation_matrix[0][2], random_rotation_matrix[1][2], random_rotation_matrix[2][2]}};

                              // compute new misfit
                              double misfit_new, rms_diff_struct_tensor_new;
                              std::tie(misfit_new, rms_diff_struct_tensor_new) = compute_misfit(axes_new, struct_tensors[composition_number], tolerance);

                              double p_acc = std::exp(std::min(misfit_old - misfit_new, 0.0));
                              if (p_acc > dist(engine))
                                {
                                  axes_old = axes_new;
                                  euler[random_index] = random_EA;
                                  misfit_old = misfit_new;
                                  rms_diff_struct_tensor_old = rms_diff_struct_tensor_new;
                                  ++n_acc;
                                }
                              else
                                {
                                  axes_new = axes_old; // revert
                                }
                              std::cout << "Iteration " << n_iter << ", misfit = " << misfit_old << ", rms_diff_struct_tensor = " << rms_diff_struct_tensor_old
                                        << ", acceptance rate = " << static_cast<double>(n_acc) / n_iter << "\r" << std::flush;
                            }

                          // --- 4. save final rotation matrices back to grains_local ---
                          for (unsigned int k = 0; k < N; ++k)
                            {
                              grains_local.rotation_matrices[k] = WorldBuilder::Utilities::euler_angles_to_rotation_matrix(
                                                                    euler[k][0], euler[k][1], euler[k][2]);
                            }

                          return grains_local;
                        }
                    }
                }
            }
          return grains_local;
        }
        WB_REGISTER_FEATURE_MANTLE_LAYER_GRAINS_MODEL(StructureTensorSampling, structure tensor sampling)
      } // namespace Grains
    } // namespace MantleLayerModels
  } // namespace Features
} // namespace WorldBuilder

