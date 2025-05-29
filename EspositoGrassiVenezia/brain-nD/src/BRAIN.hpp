#ifndef BRAIN_HPP
#define BRAIN_HPP

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
class Brain
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  class ExtracellularDiffusion : public Function<dim>
  {
  public:
    ExtracellularDiffusion(const std::vector<double> &d_ext_) : d_ext(d_ext_) {};

    // void set_material_id_map(const std::map<CellId, unsigned int> &material_id_map_)
    // {
    //   material_id_map = material_id_map_;
    // }

    // void set_dof_handler(const DoFHandler<dim> &dof_handler_)
    // {
    //   dof_handler = &dof_handler_;
    // };

    // Evaluation.
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 1)
        return d_ext[0]; // TODO, GREY MATTER
      else if (component == 2)
        return d_ext[1]; // TODO, WHITE MATTER
      else
        return 0;
    }

  protected:
    // const DoFHandler<dim> *dof_handler=nullptr;
    const std::vector<double> d_ext;
    const std::vector<double> d_axn;
    // std::map<CellId, unsigned int> material_id_map;
  };

  class AxonalTransport : public Function<dim>
  {
  public:
     AxonalTransport(const std::vector<double> &d_axn_) : d_axn(d_axn_) {};

    // void set_material_id_map(const std::map<CellId, unsigned int> &material_id_map_)
    // {
    //   material_id_map = material_id_map_;
    // }

    // void set_dof_handler(const DoFHandler<dim> &dof_handler_)
    // {
    //   dof_handler = &dof_handler_;
    // };

    // Evaluation.
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 1)
        return d_axn[0]; // TODO, GREY MATTER
      else if (component == 2)
        return d_axn[1]; // TODO, WHITE MATTER
      else
        return 0;
    }

  protected:
    // const DoFHandler<dim> *dof_handler=nullptr;
    const std::vector<double> d_ext;
    const std::vector<double> d_axn;
    // std::map<CellId, unsigned int> material_id_map;
  };

  class ReactionCoefficient : public Function<dim>
  {
  public:
    ReactionCoefficient(const std::vector<double> &alpha_) : alpha(alpha_) {};

    // void set_material_id_map(const std::map<CellId, unsigned int> &material_id_map_)
    // {
    //   material_id_map = material_id_map_;
    // }

    // void set_dof_handler(const DoFHandler<dim> &dof_handler_)
    // {
    //   dof_handler = &dof_handler_;
    // };

    // Evaluation.
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 1)
        return alpha[0]; // TODO, GREY MATTER
      else if (component == 2)
        return alpha[1]; // TODO, WHITE MATTER
      else
        return 0;
    }

  protected:
    // const DoFHandler<dim> *dof_handler=nullptr;
    const std::vector<double> alpha;
    // std::map<CellId, unsigned int> material_id_map;
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

  // Neumann boundary function
  class FunctionH : public Function<dim>
  {
  public:
    FunctionH() {}

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
      /*
        Measure limits on paraview
        -63.8802 to 66.974 (delta: 130.854)
        -107.998 to 61.8228 (delta: 169.82)
        -57.3737 to 80.4985 (delta: 137.872)
      */
      if (p[0] >= -5.0 && p[0] <= 5.0 &&  
          p[1] >= -15.0 && p[1] <= -10.0 && 
          p[2] >= -45.0 && p[2] <= -40.0)  
      {
        return 0.1;
      }
      else
      {
        return 0.0;
      }
      /*if (p[1] >= -25.0 && p[1] <= -22.0)
        return 0.1;
      else
        return 0.0;
        */
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Brain(const std::string &mesh_file_name_,
        const unsigned int &r_,
        const double &T_,
        const double &deltat_,
        const double &theta_,
        const std::vector<double> &d_ext_,
        const std::vector<double> &d_axn_,
        const std::vector<double> &alpha_)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), 
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), 
        pcout(std::cout, mpi_rank == 0), 
        mesh(MPI_COMM_WORLD), 
        r(r_), 
        deltat(deltat_), 
        theta(theta_), 
        T(T_), 
        d_ext(d_ext_), 
        d_axn(d_axn_), 
        alpha(alpha_), 
        d_ext_func(d_ext_), 
        d_axn_func(d_axn_), 
        reaction_coefficient(alpha_), 
        mesh_file_name(mesh_file_name_),
        time(0.0),
        dof_handler(mesh)
  {
  }

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

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Final time.
  const double T;

  // Physical parameters
  std::vector<double> const d_ext;
  std::vector<double> const d_axn;
  std::vector<double> const alpha;

  ExtracellularDiffusion d_ext_func;
  AxonalTransport d_axn_func;

  ReactionCoefficient reaction_coefficient;

  // Mesh file name.
  const std::string mesh_file_name;

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

  //material id map
  std::map<CellId, unsigned int> material_id_map;
};

#endif