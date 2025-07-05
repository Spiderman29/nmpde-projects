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
#include "params.hpp"

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

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 1)
        return d_ext[0];
      else if (component == 2)
        return d_ext[1];
      else
        return 0;
    }

  protected:
    const std::vector<double> d_ext;
  };

  class AxonalTransport : public Function<dim>
  {
  public:
    AxonalTransport(const std::vector<double> &d_axn_,
                        const Point<dim> &center_,
                        const double r_interface_,
                        const double steepness_ = 10.0)
      : Function<dim>(),
        d_axn(d_axn_),
        center(center_),
        r_interface(r_interface_),
        k(steepness_) {}

    // Evaluation.
    virtual double value(const Point<dim> &p,
                       const unsigned int material_id) const override
    {
      if (material_id == 1)
        return d_axn[0];
      else if (material_id == 2)
      {
        const double r = (p - center).norm(); // distance from the center
        const double d = r - r_interface; // distance from the interface
        const double s = 1.0 / (1.0 + std::exp(-k * d));

        return d_axn[0] * (1.0 - s) + d_axn[1] * s;
      }
      else
        return 0.0;
    }

  protected:
    const std::vector<double> d_axn;
    const Point<dim> center;
    const double r_interface;
    const double k;
  };

  class ReactionCoefficient : public Function<dim>
  {
  public:
    ReactionCoefficient(const std::vector<double> &alpha_) : alpha(alpha_) {};

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 1)
        return alpha[0];
      else if (component == 2)
        return alpha[1];
      else
        return 0;
    }

  protected:
    const std::vector<double> alpha;
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

  // Initial condition
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual void set_center(const Point<dim> &center_)
    {
      center = center_;
    }

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {

      /*
        Measure limits on paraview
        x: -63.8802 to 66.974 (delta: 130.854)
        y: -107.998 to 61.8228 (delta: 169.82)
        z: -57.3737 to 80.4985 (delta: 137.872)
      */

      const double center_radius = 8;
      if ((p - center).norm_square() <= center_radius * center_radius)
      {
        return 0.1;
      }
      else
      {
        return 0.0;
      }
    }

  private:
    Point<dim> center;
  };

  class Diffusion
  {
  public:
    Diffusion(std::string type_of_diffusion) : str_diffusion(type_of_diffusion) {};

    void set_center(const Point<dim> &center_)
    {
      center = center_;
    }

    Tensor<1, dim> compute_direction(const Point<dim> &p, const unsigned int material_id) const
    {
      if (str_diffusion == "radial")
      {
        return radial_direction(p);
      }
      else if (str_diffusion == "circumferential")
      {
        return circumferential_direction(p);
      }
      else if (str_diffusion == "axonal")
      {
        return diffusion(p, material_id);
      }
      else
      {
        throw std::invalid_argument("Unknown axonal direction type");
      }
    }

  private:
    std::string str_diffusion;

    Point<dim> center;

    Tensor<1, dim> radial_direction(const Point<dim> &p) const
    {
      Tensor<1, dim> normal;
      Tensor<1, dim> radial_direction;
      radial_direction = p - center;
      double radius = radial_direction.norm();
      if (radius > 1e-10)
      {
        normal = radial_direction / radius;
      }
      else
      {
        // If at center point, use default direction
        normal[0] = 1.0;
        for (unsigned int d = 1; d < dim; ++d)
          normal[d] = 0.0;
      }
      return normal;
    }

    Tensor<1, dim> circumferential_direction(const Point<dim> &p) const
    {
      Tensor<1, dim> normal;
      Tensor<1, dim> circumferential;
      circumferential = p - center;
      double norm = circumferential.norm();
      if(norm > 1e-12) circumferential/=norm;
      normal[0] = circumferential[1];
      normal[1] = -circumferential[0];
      if (dim == 3) normal[2] = 0.0;
      norm = normal.norm();
      if (norm > 1e-12) normal/=norm;

      return normal;

    }

    Tensor<1, dim> diffusion(const Point<dim> &p, const unsigned int material_id) const
    {
      if (material_id == 1)
      {
        return circumferential_direction(p);
      }
      else if (material_id == 2)
      {
        return radial_direction(p);
      }
      else
      {
        return Tensor<1, dim>({0.0, 0.0, 0.0});
      }
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Brain(const Params p)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        pcout(std::cout, mpi_rank == 0),
        mesh(MPI_COMM_WORLD),
        r(p.degree),
        deltat(p.deltat),
        theta(p.theta),
        T(p.T),
        d_ext(p.d_ext),
        d_axn(p.d_axn),
        alpha(p.alpha),
        d_ext_func(p.d_ext),
        //d_axn_func(p.d_axn),
        type_of_diffusion(p.diffusion),
        diffusion(p.diffusion),
        reaction_coefficient(p.alpha),
        mesh_file_name(p.mesh_file_name),
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

  void
  compute_material_mapping();

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
  std::shared_ptr<Function<dim>> d_axn_func;
  std::string type_of_diffusion;
  Diffusion diffusion;

  ReactionCoefficient reaction_coefficient;

  // Mesh file name.
  const std::string mesh_file_name;

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

  //Material vector
  TrilinosWrappers::MPI::Vector material_vector;

  // material id map
  std::map<CellId, unsigned int> material_id_map;
};

#endif