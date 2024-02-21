/*
  Copyright (C) 2018 - 2021 by the authors of the World Builder code.

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

#include "world_builder/features/plume_models/temperature/uniform.h"


#include "world_builder/nan.h"
#include "world_builder/types/array.h"
#include "world_builder/types/double.h"
#include "world_builder/types/object.h"
#include "world_builder/types/one_of.h"
#include "world_builder/types/value_at_points.h"
#include "world_builder/utilities.h"

#include "world_builder/kd_tree.h"

namespace WorldBuilder
{

  using namespace Utilities;

  namespace Features
  {
    namespace PlumeModels
    {
      namespace Temperature
      {
        Uniform::Uniform(WorldBuilder::World *world_)
          :
          min_depth(NaN::DSNAN),
          max_depth(NaN::DSNAN),
          temperature(NaN::DSNAN),
          operation(Operations::REPLACE)
        {
          this->world = world_;
          this->name = "uniform";
        }

        Uniform::~Uniform()
          = default;

        void
        Uniform::declare_entries(Parameters &prm, const std::string & /*unused*/)
        {
          // Document plugin and require entries if needed.
          // Add `temperature` to the required parameters.
          prm.declare_entry("", Types::Object({"temperature"}),
                            "Uniform temperature model. Set the temperature to a constant value.");

          // Declare entries of this plugin
          prm.declare_entry("min depth", Types::Double(0),
                            "The depth in meters from which the temperature of this feature is present.");
          prm.declare_entry("max depth", Types::Double(std::numeric_limits<double>::max()),
                            "The depth in meters to which the temperature of this feature is present.");

          prm.declare_entry("temperature", Types::Double(293.15),
                            "The temperature in degree Kelvin which this feature should have");

        }

        void
        Uniform::parse_entries(Parameters &prm)
        {
          min_depth = prm.get<double>("min depth");
          max_depth = prm.get<double>("max depth");
          operation = string_operations_to_enum(prm.get<std::string>("operation"));
          temperature = prm.get<double>("temperature");
        }


        double
        Uniform::get_temperature(const Point<3> & /*position_in_cartesian_coordinates*/,
                                 const Objects::NaturalCoordinate & /*position_in_natural_coordinates*/,
                                 const double depth,
                                 const double  /*gravity*/,
                                 double temperature_,
                                 const double /*feature_min_depth*/,
                                 const double /*feature_max_depth*/) const
        {
          if (depth <= max_depth && depth >= min_depth)
            {
              return apply_operation(operation,temperature_,temperature);
            }
          return temperature_;
        }

        WB_REGISTER_FEATURE_PLUME_TEMPERATURE_MODEL(Uniform, uniform)
      } // namespace Temperature
    } // namespace PlumeModels
  } // namespace Features
} // namespace WorldBuilder
