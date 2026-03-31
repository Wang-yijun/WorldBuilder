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

#ifndef WORLD_BUILDER_FEATURES_MANTLE_LAYER_MODELS_GRAINS_STRUCTURE_TENSOR_SAMPLING_H
#define WORLD_BUILDER_FEATURES_MANTLE_LAYER_MODELS_GRAINS_STRUCTURE_TENSOR_SAMPLING_H

#include "world_builder/features/mantle_layer_models/grains/interface.h"
#include "world_builder/objects/surface.h"

#include <random>

namespace WorldBuilder
{
  namespace Features
  {
    namespace MantleLayerModels
    {
      namespace Grains
      {
        class StructureTensorSampling final : public Interface
        {
          public:
            StructureTensorSampling(WorldBuilder::World *world);
            ~StructureTensorSampling() override final;

            static void declare_entries(Parameters &prm,
                                        const std::string &parent_name = "");

            void parse_entries(Parameters &prm,
                               const std::vector<Point<2>> &coordinates) override final;

            std::pair<double,double>
            compute_misfit(const std::array<std::vector<std::array<double,3>>,3> &axes,
                           const std::array<std::array<std::array<double,3>,3>,3> &a2_data,
                           double sigma) const;

            // Compute the second-order structure tensor from a set of axes
            // axes: vector of 3D vectors representing grain axes
            // weights: optional weights for each vector
            // Returns: 3x3 second-order structure tensor
            std::array<std::array<double,3>,3>
            struct_tensor_from_axes(const std::vector<std::array<double,3>> &axes,
                                    const std::vector<double> &weights = std::vector<double>()) const;

            WorldBuilder::grains
            get_grains(const Point<3> &position,
                       const Objects::NaturalCoordinate &position_in_natural_coordinates,
                       const double depth,
                       const unsigned int composition_number,
                       WorldBuilder::grains grains,
                       const double feature_min_depth,
                       const double feature_max_depth) const override final;

          private:
            // uniform grains submodule parameters
            double min_depth;
            Objects::Surface min_depth_surface;
            double max_depth;
            Objects::Surface max_depth_surface;
            std::vector<unsigned int> grains;
            std::vector<unsigned int> compositions;
            std::vector<std::array<std::array<std::array<double,3>,3>,3>> struct_tensors;
            unsigned int n_grains;
            double tolerance;
            unsigned int max_iterations;


        };
      }
    }
  }
}

#endif