////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch.hpp>

#include "lbann/utils/argument_parser.hpp"

#include "lbann/utils/environment_variable.hpp"

#include "stubs/preset_env_accessor.hpp"

SCENARIO ("Testing the argument parser", "[parser][utilities]")
{
  GIVEN ("An argument parser")
  {
    lbann::utils::argument_parser parser;
    WHEN ("The default arguments are passed")
    {
      int const argc = 1;
      char const* argv[] = { "argument_parser_test.exe" };
      THEN ("The parser recognizes the executable name")
      {
        REQUIRE_NOTHROW(parser.parse(argc, argv));
        REQUIRE(
          parser.get_exe_name() == "argument_parser_test.exe");
      }
    }
    WHEN ("The short help flag is passed")
    {
      int const argc = 2;
      char const* argv[] = {"argument_parser_test.exe", "-h"};
      THEN ("The parser notes that help has been requested.")
      {
        REQUIRE_FALSE(parser.help_requested());
        REQUIRE_NOTHROW(parser.parse(argc, argv));
        REQUIRE(parser.help_requested());
      }
    }
    WHEN ("The long help flag is passed")
    {
      int const argc = 2;
      char const* argv[argc] = {"argument_parser_test.exe", "--help"};
      THEN ("The parser notes that help has been requested.")
      {
        REQUIRE_NOTHROW(parser.parse(argc, argv));
        REQUIRE(parser.help_requested());
      }
    }
    WHEN ("A boolean flag is added")
    {
      auto verbose =
        parser.add_flag(
          "verbose", {"-v", "--verbose"}, "print verbosely");
      THEN ("The flag's option name is known")
      {
        REQUIRE(parser.option_is_defined("verbose"));
        REQUIRE_FALSE(verbose);
      }
      AND_WHEN("The flag is passed")
      {
        int const argc = 2;
        char const* argv[]
          = {"argument_parser_test.exe", "--verbose"};
        REQUIRE_FALSE(parser.get<bool>("verbose"));
        THEN ("The verbose flag is registered")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.get<bool>("verbose"));
          REQUIRE(verbose);
        }
      }
    }
    WHEN ("An option is added")
    {
      auto num_threads =
        parser.add_option("number of threads", {"-t", "--num_threads"},
                          "The number of threads to use in this test.", 1);
      THEN ("The option is registered with the parser.")
      {
        REQUIRE(parser.option_is_defined("number of threads"));
        REQUIRE(parser.template get<int>("number of threads") == 1);
        REQUIRE(num_threads == 1);
      }
      AND_WHEN ("The short option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[] = {"argument_parser_test.exe", "-t", "9"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<int>("number of threads") == 9);
          REQUIRE(num_threads == 9);
        }
      }
      AND_WHEN ("The long option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[]
          = {"argument_parser_test.exe", "--num_threads", "13"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<int>("number of threads") == 13);
          REQUIRE(num_threads == 13);
        }
      }
    }
    WHEN ("A string-valued option is added")
    {
      auto name =
        parser.add_option("my name", {"-n", "--name", "--my_name"},
                          "The number of threads to use in this test.",
                          "<unregistered name>");
      THEN ("The option is registered with the parser.")
      {
        REQUIRE(parser.option_is_defined("my name"));
        REQUIRE(parser.template get<std::string>("my name")
                == "<unregistered name>");
      }
      AND_WHEN ("The short option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[]
          = {"argument_parser_test.exe", "-n", "Banana Joe"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<std::string>("my name")
            == "Banana Joe");
          REQUIRE(name == "Banana Joe");
        }
      }
      AND_WHEN ("The first long option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[]
          = {"argument_parser_test.exe", "--name", "Plantain Pete"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<std::string>("my name")
            == "Plantain Pete");
          REQUIRE(name == "Plantain Pete");
        }
      }
      AND_WHEN ("The second long option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[]
          = {"argument_parser_test.exe", "--my_name",
             "Jackfruit Jill"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<std::string>("my name")
            == "Jackfruit Jill");
          REQUIRE(name == "Jackfruit Jill");
        }
      }
    }

    WHEN ("A required argument is added")
    {
      auto required_int =
        parser.add_required_argument<int>(
        "required", "This argument is required.");
      THEN ("The option is recognized")
      {
        REQUIRE(parser.option_is_defined("required"));
      }
      AND_WHEN("The option is not passed in the arguments")
      {
        int const argc = 1;
        char const* argv[argc] = {"argument_parser_test.exe"};

        THEN ("Finalization fails.")
        {
          parser.parse_no_finalize(argc,argv);
          REQUIRE_THROWS_AS(
            parser.finalize(),
            lbann::utils::argument_parser::missing_required_arguments);
        }
      }
      AND_WHEN("The option is passed in the arguments")
      {
        int const argc = 2;
        char const* argv[argc] = {"argument_parser_test.exe","13"};

        THEN ("Parsing is successful and the value is updated.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(required_int == 13);
        }
      }
      AND_WHEN("Another is added option and passed in the arguments")
      {
        auto required_string =
          parser.add_required_argument<std::string>(
            "required string", "This argument is also required.");

        int const argc = 3;
        char const* argv[argc] = {"argument_parser_test.exe","13","bananas"};

        THEN ("Parsing is successful and the values are updated.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(required_int == 13);
          REQUIRE(required_string == "bananas");
        }
      }
    }

    WHEN ("An optional argument is added")
    {
      auto optional_int =
        parser.add_argument(
          "optional", "This argument is optional.", -1);
      THEN ("The option is recognized")
      {
        REQUIRE(parser.option_is_defined("optional"));
        REQUIRE(parser.template get<int>("optional") == -1);
        REQUIRE(optional_int == -1);
      }
      AND_WHEN("The option is not passed in the arguments")
      {
        int const argc = 1;
        char const* argv[argc] = {"argument_parser_test.exe"};

        THEN ("Parsing succeeds with no update to the value.")
        {
          REQUIRE_NOTHROW(parser.parse(argc,argv));
          REQUIRE(parser.template get<int>("optional") == -1);
          REQUIRE(optional_int == -1);
        }
      }
      AND_WHEN("The option is passed in the arguments")
      {
        int const argc = 2;
        char const* argv[argc] = {"argument_parser_test.exe","13"};

        THEN ("Parsing is successful and the value is updated.")
        {
          REQUIRE_NOTHROW(parser.parse(argc,argv));
          REQUIRE(parser.template get<int>("optional") == 13);
          REQUIRE(optional_int == 13);
        }
      }
      AND_WHEN("Another argument is added and passed in the arguments")
      {
        auto optional_string =
          parser.add_argument(
            "optional string", "This argument is also optional.",
            "pickles");

        int const argc = 3;
        char const* argv[argc] = {"argument_parser_test.exe","42","bananas"};

        THEN ("Parsing is successful and the values are updated.")
        {
          REQUIRE(optional_int == -1);
          REQUIRE(optional_string == "pickles");
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(optional_int == 42);
          REQUIRE(optional_string == "bananas");
        }
      }
      AND_WHEN("A required argument is added and passed in the arguments")
      {
        auto required_string =
          parser.add_required_argument<std::string>(
            "required string", "This argument is required.");

        AND_WHEN("The arguments are passed in the add order")
        {
          int const argc = 3;
          char const* argv[argc] = {
            "argument_parser_test.exe","42","bananas"};
          THEN ("Parsing fails because required must come first")
          {
            REQUIRE_THROWS(parser.parse(argc,argv));
            REQUIRE(required_string == "42");
          }
        }
        AND_WHEN("The arguments are passed in the right order")
        {
          int const argc = 3;
          char const* argv[argc] = {
            "argument_parser_test.exe","bananas","42"};
          THEN ("The arguments must be reversed")
          {
            REQUIRE(optional_int == -1);
            REQUIRE_NOTHROW(parser.parse(argc, argv));
            REQUIRE(optional_int == 42);
            REQUIRE(required_string == "bananas");
          }
        }
      }
    }

    WHEN ("A flag with env variable override is added")
    {
      using namespace lbann::utils::stubs;
      using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

      auto verbose =
        parser.add_flag("verbose", {"-v"},
                        TestENV("VALUE_IS_TRUE"), "");

      THEN("The flag registers as true.")
      {
        REQUIRE(parser.option_is_defined("verbose"));
        REQUIRE(verbose);
      }

      AND_WHEN ("The flag is passed on the command line")
      {
        int const argc = 2;
        char const* argv[]
          = {"argument_parser_test.exe", "-v"};

        THEN ("The verbose flag is registered")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.get<bool>("verbose"));
          REQUIRE(verbose);
        }
      }
    }

    WHEN ("A flag with false-valued env variable override is added")
    {
      using namespace lbann::utils::stubs;
      using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

      auto verbose =
        parser.add_flag("verbose", {"-v"},
                        TestENV("VALUE_IS_FALSE"), "");

      THEN("The flag registers as false.")
      {
        REQUIRE(parser.option_is_defined("verbose"));
        REQUIRE_FALSE(verbose);
      }

      AND_WHEN ("The flag is passed on the command line")
      {
        int const argc = 2;
        char const* argv[]
          = {"argument_parser_test.exe", "-v"};

        THEN ("The verbose flag is registered")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.get<bool>("verbose"));
          REQUIRE(verbose);
        }
      }
    }

    WHEN ("A flag with false-valued env variable override is added")
    {
      using namespace lbann::utils::stubs;
      using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

      auto verbose =
        parser.add_flag("verbose", {"-v"},
                        TestENV("VALUE_IS_UNDEFINED"), "");

      THEN("The flag registers as false.")
      {
        REQUIRE(parser.option_is_defined("verbose"));
        REQUIRE_FALSE(verbose);
      }

      AND_WHEN ("The flag is passed on the command line")
      {
        int const argc = 2;
        char const* argv[]
          = {"argument_parser_test.exe", "-v"};

        THEN ("The verbose flag is registered")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.get<bool>("verbose"));
          REQUIRE(verbose);
        }
      }
    }

    WHEN ("A defined environment varible is added")
    {
      using namespace lbann::utils::stubs;
      using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

      parser.add_option(
        "apple", {"-a"}, TestENV("APPLE"),
        "Apple pie tastes good.", 1.23);

      REQUIRE(parser.option_is_defined("apple"));

      AND_WHEN("The option is not passed in the arguments")
      {
        int const argc = 1;
        char const* argv[argc] = {"argument_parser_test.exe"};

        THEN("The option has the value defined in the environment")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.template get<double>("apple") == 3.14);
        }
      }

      AND_WHEN("The option is passed in the arguments")
      {
        int const argc = 3;
        char const* argv[argc] = {"argument_parser_test.exe", "-a", "5.0"};
        THEN("The option has the value defined in command line args")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.template get<double>("apple") == 5.0);
        }
      }
    }

    WHEN ("An undefined environment varible is added")
    {
      using namespace lbann::utils::stubs;
      using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

      parser.add_option(
        "platypus", {"-p"}, TestENV("DOESNT_EXIST"),
        "This variable won't exist.", 1.23);

      REQUIRE(parser.option_is_defined("platypus"));

      AND_WHEN("The option is not passed in the arguments")
      {
        int const argc = 1;
        char const* argv[argc] = {"argument_parser_test.exe"};

        THEN("The option has the default value")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.template get<double>("platypus") == 1.23);
        }
      }
      AND_WHEN("The option is passed in the arguments")
      {
        int const argc = 3;
        char const* argv[argc] = {"argument_parser_test.exe", "-p", "2.0"};
        THEN("The option has the value defined in the command line args")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.template get<double>("platypus") == 2.0);
        }
      }
    }

    WHEN ("A defined string environment varible is added")
    {
      using namespace lbann::utils::stubs;
      using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

      parser.add_option(
        "pizza", {"-p"}, TestENV("PIZZA"),
        "Mmmm pizza.", "mushroom");

      REQUIRE(parser.option_is_defined("pizza"));

      AND_WHEN("The option is not passed in the arguments")
      {
        int const argc = 1;
        char const* argv[argc] = {"argument_parser_test.exe"};

        THEN("The option has the value defined in the environment")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.template get<std::string>("pizza") == "pepperoni");
        }
      }

      AND_WHEN("The option is passed in the arguments")
      {
        int const argc = 3;
        char const* argv[argc] = {"argument_parser_test.exe", "-p", "hawaiian"};
        THEN("The option has the value defined in the command line args")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.template get<std::string>("pizza") == "hawaiian");
        }
      }
    }

    WHEN ("An undefined environment varible is added to a string option")
    {
      using namespace lbann::utils::stubs;
      using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

      parser.add_option(
        "platypus", {"-p"}, TestENV("DOESNT_EXIST"),
        "This variable won't exist.", "so cute");

      REQUIRE(parser.option_is_defined("platypus"));

      AND_WHEN("The option is not passed in the arguments")
      {
        int const argc = 1;
        char const* argv[argc] = {"argument_parser_test.exe"};

        THEN("The option has the default value")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.template get<std::string>("platypus") == "so cute");
        }
      }
      AND_WHEN("The option is passed in the arguments")
      {
        int const argc = 3;
        char const* argv[argc] = {"argument_parser_test.exe", "-p", "llama"};
        THEN("The option has the value defined in the command line args")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.template get<std::string>("platypus") == "llama");
        }
      }
    }
  }
}
