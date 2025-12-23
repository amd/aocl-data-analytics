#!/usr/bin/env python3
##Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.##
"""
CMake Preset Generator for AOCL-Data Analytics (AOCL-DA)
This script generates CMake presets for different configurations of AOCL-DA.
It supports both Windows and Linux platforms with different compilers,
integer sizes, and build types.

Usage:
  - Single preset: python preset_generator.py linux-clang-st-lp64-static-release
  - All presets: python preset_generator.py --generate-all [options]
"""
import sys
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Define all preset options with their descriptions
PRESET_OPTIONS: Dict[str, Tuple[List[str], str, str]] = {
    "platform": (["linux", "win"], "Platform options (required first element)", ""),
    "generator": (["make", "ninja", "msvc"], "Generator options", ""),
    "compiler": (["gcc", "clang"], "Compiler options", "for Linux only"),
    "threading": (["st", "mt"], "Threading options", "st (single-threaded), mt (multi-threaded)"),
    "int_size": (["lp64", "ilp64"], "Integer size options", ""),
    "lib_type": (["static", "shared"], "Library type options", ""),
    "build_python": (["python"], "Python Bindings", ""),
    "arch": (["native", "dynamic", "znver2", "znver3", "znver4", "znver5"],
              "Architecture options", ""),
    "build_type": (["release", "debug", "relwithdebinfo"], "Build type options", ""),
    "sanitizer": (["asan"], "Address Sanitizer", ""),
    "mem_sanitizer": (["memsan"], "Memory Sanitizer", ""),
    "code_coverage": (["coverage"], "Code Coverage", ""),
    "valgrind": (["valgrind"], "Valgrind Support", ""),
    "documentation": (["doc"], "Documentation", ""),
    "libmem": (["libmem"], "Memory Library", "")
}

def print_available_options():
    """Print all available options for preset configuration"""
    print("Available preset options:")

    for _, (options, description, notes) in PRESET_OPTIONS.items():
        print(f"\n{description}:")
        print(f"  {', '.join(options)}", end='')
        if notes:
            print(f" ({notes})")
        else:
            print()

    print("\nUsage examples:")
    print("  Linux with compiler:")
    print("    python preset_generator.py linux-make-gcc-st-lp64-znver3-static")
    print("  Windows:")
    print("    python preset_generator.py win-ninja-st-lp64-znver3-static")
    print("  Generate all presets:")
    print("    python preset_generator.py --generate-all [--dry-run] [--limit N]")

    print("\nNotes:")
    print("- Options should be separated by hyphens (-)")
    print("- Each category can only have one option")
    print("- For Windows, compiler options are not allowed")

def generate_build_dir_name(components: Dict[str, str]) -> str:
    """
    Generate a build directory name from components following the order in PRESET_OPTIONS.
    Only includes non-empty components.
    """
    parts = []
    # Add components in the order they appear in PRESET_OPTIONS
    for category in PRESET_OPTIONS:
        if components[category] != "empty":
            if category == "generator" and components["platform"] == "win":
                continue
            parts.append(components[category])

    # Join parts with hyphens
    return "-".join(parts)

def generate_install_dir_name(components: Dict[str, str]) -> str:
    """
    Generate an install directory name from components following the order in PRESET_OPTIONS.
    Only includes non-empty components and skips specified categories.
    """
    # Categories to skip in the install directory name
    skip_categories = ["int_size", "lib_type", "build_python", "documentation",
                      "generator", "threading"]

    parts = []
    # Add components in the order they appear in PRESET_OPTIONS
    for category in PRESET_OPTIONS:
        if category in skip_categories:
            continue
        if components[category] != "empty":
            parts.append(components[category])

    # Join parts with hyphens
    return "-".join(parts)

def parse_preset_name(preset_name):
    """
    Parse preset name into components and return a dictionary with all options.
    Each option not found in the preset name will have value "empty".
    """
    parts = preset_name.split('-')
    if not parts:
        raise ValueError("Empty preset name")
    # Initialize result dictionary with empty values
    result = {category: "empty" for category in PRESET_OPTIONS}

    # Check each part against all options
    for part in parts:
        matches = []
        for category, (options, _, _) in PRESET_OPTIONS.items():
            if part in options:
                matches.append((category, part))
        # Check if this part matches multiple categories
        if len(matches) > 1:
            categories = [m[0] for m in matches]
            raise ValueError(f"Part '{part}' matches multiple categories: {categories}")
        # If we found exactly one match, update the result
        if len(matches) == 1:
            category, value = matches[0]
            if result[category] != "empty":
                raise ValueError(f"Multiple values for category '{category}' found: \
                                 {result[category]} and {value}")
            result[category] = value
        # If no matches found, raise an error
        else:
            raise ValueError(f"Part '{part}' does not match any category")

    # For Windows platform, always add msvc as a generator but do not add it in the preset name
    if result["platform"] == "win":
        result["generator"] = "msvc"

    # Validate Windows platform cannot have compiler specified
    if result["platform"] == "win" and result["compiler"] != "empty":
        raise ValueError("Compiler options (gcc, clang) cannot be used with Windows platform")

    # Validate that Python bindings require shared libraries
    if result["lib_type"] == "static" and result["build_python"] == "python":
        raise ValueError("Python bindings (python) require shared libraries. "
                         "Please use 'shared' instead of 'static'")

    return result

def generate_preset(preset_name: str) -> Dict:
    """Generate preset configuration based on name components"""
    try:
        components = parse_preset_name(preset_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nValid format examples:")
        print("  Linux: python preset_generator.py linux-make-clang-st-python")
        print("  Windows: python preset_generator.py win-ninja-st-lp64-znver3-static")
        sys.exit(1)
    # Build inherits list with all non-empty components
    inherits = ["base"]
    for _, value in components.items():
        if value != "empty":
            inherits.append(value)

    # Generate build directory name based on components
    build_dir_name = generate_build_dir_name(components)

    # Generate install directory name based on filtered components
    install_dir_name = generate_install_dir_name(components)

    # Create the preset structure
    preset = {
        "version": 6,
        "include": ["base.json"],
        "configurePresets": [
            {
                "name": preset_name,
                "inherits": inherits,
                "cacheVariables": {
                    "CMAKE_INSTALL_PREFIX": "${sourceDir}/install-" + install_dir_name
                },
                "binaryDir": "${sourceDir}/build-" + build_dir_name,
                "hidden": False
            }
        ],
        "buildPresets": [
            {
                "name": preset_name,
                "configurePreset": preset_name,
                "inherits": "base"
            },
            {
                "name": preset_name+"-install",
                "configurePreset": preset_name,
                "inherits": "base",
                "targets": "install"
            }
        ],
        "testPresets": [
            {
                "name": preset_name,
                "configurePreset": preset_name,
                "inherits": "base"
            }
        ],
        "workflowPresets": [
            {
                "name": preset_name,
                "description": "Build and check AOCL-DA",
                "steps": [
                    {
                        "type": "configure",
                        "name": preset_name
                    },
                    {
                        "type": "build",
                        "name": preset_name,
                        "name": preset_name+"-install"
                    },
                    {
                        "type": "test",
                        "name": preset_name
                    }
                ]
            }
        ]
    }
    return preset

def update_includes_file(preset_name):
    """Update includes.json with the new preset file if not already present"""
    script_dir = Path(__file__).parent
    includes_file = script_dir / "includes.json"
    # Create includes.json if it doesn't exist
    if not includes_file.exists():
        includes_data = {"include": []}
        includes_file.write_text(json.dumps(includes_data, indent=2))
    # Read current includes
    with includes_file.open('r') as f:
        includes_data = json.load(f)
    # Add new preset file if not already in the list
    preset_file = f"{preset_name}.json"
    if preset_file not in includes_data["include"]:
        includes_data["include"].append(preset_file)
        # Write back updated includes
        with includes_file.open('w') as f:
            json.dump(includes_data, f, indent=2)
        print(f"Added {preset_file} to includes.json")
    else:
        print(f"{preset_file} already in includes.json")

def single_preset(preset_name):
    """Process a single preset (original preset_generator.py functionality)"""
    preset = generate_preset(preset_name)
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    output_file = script_dir / f"{preset_name}.json"

    with output_file.open('w') as f:
        json.dump(preset, f, indent=2)
    print(f"Generated preset configuration in {output_file}")
    # Update includes.json
    update_includes_file(preset_name)
    return output_file

def generate_all_presets(dry_run=False, limit=None):
    """Generate all preset combinations (from generate_all_presets.py)"""
    # Define the options for each category
    platforms = ["linux", "win"]
    compilers = ["gcc", "clang"]  # Only for Linux
    threading = ["st", "mt"]
    int_size = ["lp64", "ilp64"]
    lib_type = ["static", "shared"]
    build_type = ["release", "debug"]
    arch = ["native", "dynamic"]
    build_python = ["", "python"]  # Empty string means no python

    total_presets = 0
    successful_presets = 0
    skipped_presets = 0

    # Process each platform separately due to compiler constraints
    for platform in platforms:
        if platform == "linux":
            # Linux presets - with compilers
            for comp, thread, int_s, lib, btype, ar, python in itertools.product(
                compilers, threading, int_size, lib_type, build_type, arch, build_python
            ):
                # Skip invalid combination: static library with Python bindings
                if lib == "static" and python == "python":
                    skipped_presets += 1
                    continue

                preset_parts = [platform, comp, thread, int_s, lib, btype, ar]
                if python:
                    preset_parts.append(python)
                preset_name = "-".join(filter(None, preset_parts))

                total_presets += 1
                if limit and total_presets > limit:
                    break

                print(f"[{total_presets}] Generating preset: {preset_name}")
                if not dry_run:
                    try:
                        single_preset(preset_name)
                        successful_presets += 1
                    except Exception as e:
                        print(f"  Error: {str(e)}")

        else:  # Windows presets - no compilers
            for thread, int_s, lib, btype, ar, python in itertools.product(
                threading, int_size, lib_type, build_type, arch, build_python
            ):
                # Skip invalid combination: static library with Python bindings
                if lib == "static" and python == "python":
                    skipped_presets += 1
                    continue

                preset_parts = [platform, thread, int_s, lib, btype, ar]
                if python:
                    preset_parts.append(python)
                preset_name = "-".join(filter(None, preset_parts))

                total_presets += 1
                if limit and total_presets > limit:
                    break

                print(f"[{total_presets}] Generating preset: {preset_name}")
                if not dry_run:
                    try:
                        single_preset(preset_name)
                        successful_presets += 1
                    except Exception as e:
                        print(f"  Error: {str(e)}")

        if limit and total_presets >= limit:
            print(f"Reached limit of {limit} presets")
            break

    print(f"\nGenerated {successful_presets} of {total_presets} presets successfully.")
    print(f"Skipped {skipped_presets} invalid combinations (static lib with Python).")
    if dry_run:
        print("This was a dry run. No presets were actually generated.")

def main():
    # Custom help formatter class
    class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def format_help(self):
            # Start with the standard help text
            help_text = super().format_help()

            # We'll capture the output of print_available_options() to a string
            import io
            from contextlib import redirect_stdout

            options_output = io.StringIO()
            with redirect_stdout(options_output):
                print_available_options()

            # Combine both help texts
            return help_text + "\n\n" + options_output.getvalue()

    parser = argparse.ArgumentParser(
        description="CMake Preset Generator for AOCL-Data Analytics (AOCL-DA)",
        formatter_class=CustomHelpFormatter
    )
    parser.add_argument("preset_name", nargs="?",
                        help="Preset name (not required with --generate-all)")
    parser.add_argument("--generate-all", action="store_true",
                        help="Generate all preset combinations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing (with --generate-all)")
    parser.add_argument("--limit", type=int,
                        help="Limit number of presets to generate (with --generate-all)")
    args = parser.parse_args()

    print("CMake Preset Generator for AOCL-Data Analytics (AOCL-DA)")

    if args.generate_all:
        # Execute the generate-all functionality
        generate_all_presets(args.dry_run, args.limit)
    elif args.preset_name:
        # Execute the single preset generation (original functionality)
        single_preset(args.preset_name)
    else:
        print_available_options()
        sys.exit(1)

if __name__ == "__main__":
    main()
