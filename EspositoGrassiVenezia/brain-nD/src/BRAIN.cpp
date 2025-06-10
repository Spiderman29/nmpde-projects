#include "BRAIN.hpp"

void Brain::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    int count_1 = 0;
    int count_2 = 0;
    int count_3 = 0;

    for (const auto &cell : mesh.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        material_id_map[cell->id()] = cell->material_id();
        switch (cell->material_id())
        {
        case 1:
          count_1++;
          break;
        case 2:
          count_2++;
          break;
        case 3:
          count_3++;
          break;
        default:
          break;
        }
      }
    }

    pcout << "Grey Matter: " << count_1 << std::endl;
    pcout << "White Matter: " << count_2 << std::endl;
    pcout << "Cerebellum: " << count_3 << std::endl;

    // diffusion_coefficient.set_material_id_map(material_id_map);
    // reaction_coefficient.set_material_id_map(material_id_map);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    jacobian_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_old = solution;
  }
}

void Brain::assemble_system()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors |
                                       update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  Tensor<2, dim> d_ext_matrix;
  Tensor<2, dim> normal_matrix;
  Tensor<1, dim> normal;
  Point<dim> center;
  // Get mesh bounding box to set center
  auto bbox = GridTools::compute_bounding_box(mesh);
  auto boundary_points = bbox.get_boundary_points();
  for (unsigned int i = 0; i < dim; ++i)
    center[i] = (boundary_points.first[i] + boundary_points.second[i]) / 2.0;
  
  AxonalDirection axonal_direction(type_of_diffusion);
  axonal_direction.set_center(center);
  std::vector<double> solution_loc(n_q);
  std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);
  std::vector<double> solution_old_loc(n_q);

  forcing_term.set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_matrix = 0.0;
    cell_residual = 0.0;
    unsigned int material_id = material_id_map[cell->id()];
    Tensor<1, dim> normal;
    fe_values.get_function_values(solution, solution_loc);
    fe_values.get_function_gradients(solution, solution_gradient_loc);
    fe_values.get_function_values(solution_old, solution_old_loc);
    for (unsigned int q = 0; q < n_q; ++q)
    {
      const double d_ext_loc = d_ext_func.value(fe_values.quadrature_point(q), material_id);
      const double d_axn_loc = d_axn_func.value(fe_values.quadrature_point(q), material_id);
      const double reaction_coefficient_loc = reaction_coefficient.value(fe_values.quadrature_point(q), material_id);

      // Multiply each element of the identity matrix by d_ext_loc
      for (unsigned int i = 0; i < dim; ++i)
        d_ext_matrix[i][i] = d_ext_loc;

      normal=axonal_direction.compute_direction(fe_values.quadrature_point(q), material_id);

      // Create normal_matrix as the tensor product of normal with itself
      for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < dim; ++j) {
          normal_matrix[i][j] = normal[i] * normal[j];
        }
      }

      const Tensor<2, dim> D = d_ext_matrix + d_axn_loc * normal_matrix;

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_matrix(i, j) += fe_values.shape_value(i, q) *
                               fe_values.shape_value(j, q) / deltat *
                               fe_values.JxW(q);

          cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                               D * fe_values.shape_grad(j, q) *
                               fe_values.JxW(q);

          cell_matrix(i, j) -= (reaction_coefficient_loc * (1.0 - 2.0 * solution_loc[q])) *
                               fe_values.shape_value(j, q) *
                               fe_values.shape_value(i, q) *
                               fe_values.JxW(q);
        }

        cell_residual(i) -= fe_values.shape_value(i, q) *
                            (solution_loc[q] - solution_old_loc[q]) / deltat *
                            fe_values.JxW(q);

        cell_residual(i) -= fe_values.shape_grad(i, q) *
                            D * solution_gradient_loc[q] *
                            fe_values.JxW(q);

        cell_residual(i) += reaction_coefficient_loc * solution_loc[q] * (1 - solution_loc[q]) * fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

  
    cell->get_dof_indices(dof_indices);
    jacobian_matrix.add(dof_indices, cell_matrix);
    residual_vector.add(dof_indices, cell_residual);
  }
  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);
}

void Brain::solve_linear_system()
{
  SolverControl solver_control(1000, 1e-6 * residual_vector.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
      jacobian_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));
  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
}

void Brain::solve_newton()
{
  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-6;

  unsigned int n_iter = 0;
  double residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
  {
    assemble_system();
    residual_norm = residual_vector.l2_norm();

    pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
          << " - ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::flush;

    // We actually solve the system only if the residual is larger than the
    // tolerance.
    if (residual_norm > residual_tolerance)
    {
      solve_linear_system();

      solution_owned += delta_owned;
      solution = solution_owned;
    }
    else
    {
      pcout << " < tolerance" << std::endl;
    }

    ++n_iter;
  }
}

void Brain::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
      "./", "output_" + std::to_string(dim) + "D_dt" + std::to_string(deltat) + "_alpha" + std::to_string(alpha[0]) + "_dext" + std::to_string(d_ext[0]) + "_diffusion-"+type_of_diffusion, time_step, MPI_COMM_WORLD, 3);
}

void Brain::solve()
{
  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
  {
    time += deltat;
    ++time_step;

    // Store the old solution, so that it is available for assembly.
    solution_old = solution;

    pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
          << std::fixed << time << std::endl;

    // At every time step, we invoke Newton's method to solve the non-linear
    // problem.
    solve_newton();

    output(time_step);

    pcout << std::endl;
    solution_old = solution;
  }
}