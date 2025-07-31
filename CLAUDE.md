# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local shock tracking implementation in Rust, utilizing high-order discontinuous Galerkin (DG) methods for solving hyperbolic PDEs with High Order Implicit Shock Tracking (proposed by Zahr in 2020). The solver uses space-time formulations. The solver uses autodiff, an experimental feature of Rust.

## Common Development Commands

### Build Commands
```bash
# Debug build (with optimizations enabled for better performance)
cargo build

# Release build
cargo build --release

# Run the solver
cargo run
cargo run --release
```

### Testing and Quality Control
```bash
# Run tests
cargo test

# Check code without building
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy
```

## High-Level Architecture

### Core Components

1. **Discretization Module (`src/disc/`)**
   - **Basis Functions**: Implementations for 1D Lagrange and 2D triangle/quadrilateral basis functions using Lobatto points
   - **ADER-DG Scheme**: Space-time discontinuous Galerkin implementation
   - **Mesh Management**: 1D and 2D mesh structures with adaptive refinement capabilities
   - **Shock Tracking**: Specialized algorithms for detecting and tracking shocks in Burgers and Euler equations

2. **Solver Types**
   - `Disc1dBurgers`: 1D Burgers equation solver with shock tracking
   - `Disc1dBurgers1dSpaceTime`: Space-time formulation for 1D Burgers equation
   - Space-time solvers using SQP (Sequential Quadratic Programming) methods

3. **Key Traits**
   - `SpaceTimeSolver1DScalar`: Interface for 1D scalar space-time solvers
   - `Geometric2D`: Geometric operations for 2D elements
   - `P0Solver`: Piecewise constant solution methods

4. **I/O and Visualization**
   - VTU file output for ParaView visualization
   - CSV output for solution data
   - JSON-based parameter input (`inputs/solverparam.json`)

### Solution Workflow

1. **Initialization**
   - Load solver parameters from JSON
   - Create mesh (1D or 2D triangular/quadrilateral)
   - Initialize basis functions (polynomial order specified in parameters)

2. **Time Integration**
   - Space-time DG formulation
   - Adaptive time stepping based on CFL condition
   - Shock detection and mesh adaptation

3. **Output**
   - Solutions written to VTU files in `outputs/` directory
   - Both cell-averaged and nodal solutions available

### Key Dependencies
- `faer` & `faer-ext`: Linear algebra operations
- `ndarray`: Multi-dimensional arrays
- `vtkio`: VTU file writing for visualization
- `nalgebra`: Additional linear algebra support

### Important Files
- `src/main.rs`: Entry point, currently set up for 2D triangular mesh with Burgers equation
- `inputs/solverparam.json`: Solver configuration (polynomial order, CFL, time settings)
- `src/disc/burgers1d_space_time.rs`: Main space-time solver implementation
- `src/disc/mesh/mesh2d.rs`: 2D mesh generation and management

### Notes
- The project uses Rust's nightly features (autodiff)
- Reference implementations in Fortran are available in `ref/` directory
- Autodiff-related codes are for now commented out and replaced with a finite difference approach, because autodiff makes compile very slow. They will be re-enabled in the future.