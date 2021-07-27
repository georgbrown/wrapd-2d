Parameterization output data will be sent here.

param_out.obj -- Final parameterization solution. Contains both 3D vertex coordinates and 2D UV coordinates. 

objectives.txt -- Total objective (sym. Dirichlet distortion summed over all elements) at each ADMM iteration.
accumulated_time_s.txt -- Total accumulated runtime at each ADMM iteration
flips_count.txt  -- Number of element flips/inversions at each ADMM iteration
inner_count.txt -- Number of L-BFGS iterations performed during each ADMM iteration
reweighted.txt.txt -- Identifies which ADMM iterations had a reweighting operation (1=yes, 0=no)
