#ifndef BRAIN_1D_HPP
#define BRAIN_1D_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <fstream>
#include <iostream>
#include <math.h>
#include <regex>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class Brain1D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

  class DiffusionCoefficient : public Function<dim>
  {
  public:
    DiffusionCoefficient(const double &d) : d(d) {};

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return d;
    }

    protected:
      const double d;
  };

  class ReactionCoefficient : public Function<dim>
  {
  public:
    ReactionCoefficient(const double &alpha) : alpha(alpha) {};

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return alpha;
    }
    protected:
    const double alpha;
  };

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Function for Dirichlet boundary conditions.
  class FunctionG : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  //Neumann boundary function
  class FunctionH : public Function<dim>
  {
  public:
      FunctionH(){}

      // Evaluation.
      virtual double
      value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
      {
        return 0.0;
      }
  };

  // Initial condition
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      if (p[0] == 1.0)
        return 0.1;
      else
        return 0.0;
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Brain1D(const unsigned int &N_,
    const unsigned int &r_,
    const double &T_,
    const double &deltat_,
    const double &theta_,
    const double &d_,
    const double &alpha_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0), mesh(MPI_COMM_WORLD)
        , N(N_)
        , r(r_)
        , deltat(deltat_)
        , theta(theta_)
        , T(T_)
        , d(d_)
        , alpha(alpha_)
        , diffusion_coefficient(d_)
        , reaction_coefficient(alpha_)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

protected:
  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the linear system associated to the tangent problem.
  void
  solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  parallel::fullydistributed::Triangulation<dim> mesh;

  // Problem definition. ///////////////////////////////////////////////////////

  // N+1 is the number of elements.
  const unsigned int N;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Final time.
  const double T;

  // Physical parameters
  double const d;

  double const alpha;

  DiffusionCoefficient diffusion_coefficient;

  ReactionCoefficient reaction_coefficient;

  FunctionH function_h;

  FunctionU0 u_0;

  double time;

  // Forcing term.
  ForcingTerm forcing_term;

  // Discretization. ///////////////////////////////////////////////////////////

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  TrilinosWrappers::MPI::Vector solution_old;
};

#endif