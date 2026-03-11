using ImageFiltering
using ProgressMeter



function cost_func(ImageObs, ImageTest)
    """
    Mean squared error cost function comparing each pixel's observed intensity with the calculated intensity.
    """
    
    if length(ImageObs) != length(ImageTest)
        println("Length of ImageObs: $(length(ImageObs))")
        println("Length of ImageTest: $(length(ImageTest))")
        throw(ArgumentError("ImageObs and ImageTest must have the same length."))
    end
    cost = 0.0
    for i in eachindex(ImageObs, ImageTest)
        cost += (ImageTest[i] - ImageObs[i])^2
    end
    return cost
end

function GradientofCostFunction(ImageObs, ImageTest, dI_dθo, dI_da)
    """
    Calculate the gradient of the cost function with respect to the observed image.
    The gradient is computed as the difference between the observed and calculated intensities.
    """
    
    if length(ImageObs) != length(ImageTest)
        println("Length of ImageObs: $(length(ImageObs))")
        println("Length of ImageTest: $(length(ImageTest))")
        throw(ArgumentError("ImageObs and ImageTest must have the same length."))
    end

    grad_θo = 2.0 * sum((ImageTest .- ImageObs) .* (dI_dθo))
    grad_a = 2.0 * sum((ImageTest .- ImageObs) .* (dI_da))
    return grad_θo, grad_a
end


function FiniteDifferencesθ(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h, bhspin, Rout, Rstop, data = nothing, xoff = 0.0, yoff = 0.0)
    """
    Finite differences method to calculate the intensity at each pixel in the image.
    Uses threaded per-pixel integration to minimize memory overhead.
    """
    
    # --- Pre-calculations ---
    θh = th + h
    θl = th - h

    # Calculate the fov at each dim x and y 
    fovx = DXsize / ro
    fovy = DYsize / ro

    # Unitless frequency for geodesics
    freq_unitless = freq * HPL / (ME * CL * CL) 

    # Calculate the camera positions in native coordinates
    Xcamh = MVec4(camera_position(ro, θh, phi, bhspin, Rout))
    Xcaml = MVec4(camera_position(ro, θl, phi, bhspin, Rout))
    Xcamc = MVec4(camera_position(ro, th, phi, bhspin, Rout))

    # Scales the intensity of each pixel by the real size of each pixel
    # Note: Retained for logging, even if not explicitly passed to integrate_emission! in the new pattern
    scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)
    println("scale_factor = $scale_factor")

    # --- Internal Helper Function (The New Algorithm) ---
    function trace_image(Xcamera, description)
        println("Calculating $description...")
        
        # Initialize Image buffer
        local_img = zeros(Float64, pixels_x, pixels_y)
        
        # Setup Progress Meter
        p = Progress(
            pixels_x * pixels_y; 
            desc = "Raytracing $description...", 
            showspeed = true, 
            barlen = 30
        )
        
        # Raytracing Loop
        Threads.@threads for i in 0:(pixels_x - 1)
            tid = Threads.threadid()
            for j in 0:(pixels_y - 1)
                # Initialize trajectory vector for this pixel
                traj = Vector{OfTraj}()
                sizehint!(traj, maxnstep)
                
                # Calculate Geodesic
                nstep = get_pixel(traj, i, j, Xcamera, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, bhspin, Rh, Rout, Rstop, xoff, yoff) 
                
                # Resize and Integrate
                resize!(traj, nstep)
                integrate_emission!(traj, nstep, local_img, i + 1, j + 1, freq, bhspin, data)
                
                # Update Progress
                # ProgressMeter.next!(
                #     p; 
                #     showvalues = [
                #         (:thread_id, tid), 
                #         (:pixel, "($i, $j)"), 
                #         (:total_done, "$(i*pixels_y + j)/$(pixels_x * pixels_y)")
                #     ]
                # )
            end
        end
        finish!(p)
        
        # Apply Scaling
        local_img *= freq^3
        return local_img
    end

    # --- Execution ---
    
    # 1. Calculate High Trajectory/Image
    Imageh = trace_image(Xcamh, "High (+h)")
    
    # 2. Calculate Low Trajectory/Image
    Imagel = trace_image(Xcaml, "Low (-h)")

    # 3. Calculate Finite Difference
    dI_dθo = (Imageh - Imagel) / (2 * h)

    # 4. Calculate Central Image
    Imagec = trace_image(Xcamc, "Central")

    return dI_dθo, Imagec
end


# function FiniteDifferencesTrat(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h_trat, bhspin, Rout, Rstop, dump_filepath)
#     """
#     Finite differences method for trat_large (Rhigh).
#     Optimized to compute geodesic once per pixel, integrate against all temp models, 
#     and discard trajectory immediately to save memory.
#     """

#     println("Initializing Physics Data for Finite Differences...")

#     Trat_h = trat_large + h_trat
#     Trat_l = trat_large - h_trat

#     data_h = load_data(dump_filepath, Trat_h)
#     data_l = load_data(dump_filepath, Trat_l)
#     data_c = load_data(dump_filepath, trat_large)

#     fovx = DXsize / ro
#     fovy = DYsize / ro
#     Xcam = MVec4(camera_position(ro, th, phi, bhspin, Rout))

#     freq_unitless = freq * HPL / (ME * CL * CL) 

#     scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)

#     Imageh = zeros(Float64, pixels_x, pixels_y)
#     Imagel = zeros(Float64, pixels_x, pixels_y)
#     Imagec = zeros(Float64, pixels_x, pixels_y)

#     println("Calculating Trajectories and integrating High/Low/Central simultaneously...")

#     p = Progress(pixels_x * pixels_y; desc = "Computing Pixels...", showspeed = true, barlen = 30)

#     Threads.@threads for i in 0:(pixels_x - 1)
        
#         for j in 0:(pixels_y - 1)
#             traj = Vector{OfTraj}()
#             sizehint!(traj, maxnstep)

#             nstep = get_pixel(traj, i, j, Xcam, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, bhspin, Rh, Rout, Rstop, 0.0, 0.0)
            
#             resize!(traj, nstep)
            
#             # 1. High Trat
#             integrate_emission!(traj, nstep, Imageh, i + 1, j + 1, freq, bhspin, data_h)
            
#             # 2. Low Trat
#             integrate_emission!(traj, nstep, Imagel, i + 1, j + 1, freq, bhspin, data_l)

#             # 3. Central Trat
#             integrate_emission!(traj, nstep, Imagec, i + 1, j + 1, freq, bhspin, data_c)

#         end
#     end
#     finish!(p)

#     Imageh .*= freq^3
#     Imagel .*= freq^3
#     Imagec .*= freq^3

#     dI_dRhigh = (Imageh - Imagel) / (2 * h_trat)

#     data_h = nothing
#     data_l = nothing
#     data_c = nothing

#     return dI_dRhigh, Imagec
# end

function FiniteDifferencesTrat(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h_trat, bhspin, Rout, Rstop, dump_filepath)
    """
    Finite differences method for trat_large.
    Optimized for Limited RAM: Aggressive GC and Buffer Reuse.
    """

    # --- Pre-calculations ---
    fovx = DXsize / ro
    fovy = DYsize / ro
    Xcam = MVec4(camera_position(ro, th, phi, bhspin, Rout))
    freq_unitless = freq * HPL / (ME * CL * CL) 
    scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)

    # --- Internal Helper Function ---
    function trace_variant(target_trat, description)
        println("\n=== Processing $description ===")
        
        # 1. Load Data
        println("Loading simulation data for Trat = $target_trat...")
        local_data = load_data(dump_filepath, target_trat)
        
        # 2. Initialize Image Buffer
        local_img = zeros(Float64, pixels_x, pixels_y)

        # 4. Raytracing Loop
        p = Progress(pixels_x * pixels_y; desc = "Raytracing $description...", showspeed = true, barlen = 30)
        
        Threads.@threads for i in 0:(pixels_x - 1)
            # Get the pre-allocated vector for this specific thread
            id = Threads.threadid()            
            for j in 0:(pixels_y - 1)
                traj = Vector{OfTraj}()
                sizehint!(traj, maxnstep)
                
                # Calculate Geodesic (fills the reused traj vector)
                nstep = get_pixel(traj, i, j, Xcam, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, bhspin, Rh, Rout, Rstop, 0.0, 0.0) 
                
                # Resize logic (if get_pixel doesn't set length perfectly)
                # Note: If get_pixel pushes to traj, nstep is likely just length(traj)
                if length(traj) != nstep
                    resize!(traj, nstep)
                end
                
                # Integrate
                integrate_emission!(traj, nstep, local_img, i + 1, j + 1, freq, bhspin, local_data)
                
                # ProgressMeter.next!(p) # Thread-safe in newer versions
            end
        end
        finish!(p)
        
        # 5. Clean up & Force GC
        # This is the most critical step for your kernel issues
        local_data = nothing
        thread_trajs = nothing
        
        println("Forcing Garbage Collection for $description...")
        GC.gc() # Force full garbage collection
        GC.gc() # Run twice to ensure older generations are swept
        
        # Apply scaling
        local_img .*= freq^3
        return local_img
    end

    # --- Execution: Sequential Passes ---

    # 1. Calculate High Image (+h)
    Imageh = trace_variant(trat_large + h_trat, "High (+h)")

    # 2. Calculate Low Image (-h)
    Imagel = trace_variant(trat_large - h_trat, "Low (-h)")

    # 3. Calculate Derivative
    # Create the derivative matrix
    dI_dRhigh = (Imageh .- Imagel) ./ (2 * h_trat)

    # 4. Free High/Low images immediately
    # We don't need them anymore, so free them before the Central pass
    Imageh = nothing
    Imagel = nothing
    println("Clearing High/Low buffers...")
    GC.gc()

    # 5. Calculate Central Image
    Imagec = trace_variant(trat_large, "Central")

    return dI_dRhigh, Imagec
end


function FiniteDifferences_a(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep,h, bhspin, Rout, Rstop)
    """
    Finite differences method to calculate the intensity at each pixel in the image.
    This function calculates the geodesics and the intensity at each pixel using finite differences.
    """
    ah = bhspin + h
    al = bhspin - h

    # Calculate the fov at each dim x and y 
    fovx = DXsize / ro
    fovy = DYsize / ro

    # Calculate the camera position in native coordinates
    Xcamh = MVec4(camera_position(ro, th, phi, ah, Rout))
    Xcaml = MVec4(camera_position(ro, th, phi, al, Rout))

    # Scales the intensity of each pixel by the real size of each pixel
    scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)
    # integrate_emission_flag setted to false signals that the output of the function will be the trajectory and not the Image
    trajectoryh = CalculateGeodesics(Xcamh, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, ah, Rout, Rstop);
    trajectoryl = CalculateGeodesics(Xcaml, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, al, Rout, Rstop);
    
    # Integrate the emission along the geodesics
    Imageh = IpoleGeoIntensityIntegration(trajectoryh, freq, pixels_x, pixels_y, scale_factor, ah)
    Imagel = IpoleGeoIntensityIntegration(trajectoryl, freq, pixels_x, pixels_y, scale_factor, al)

    #deallocate trajectories to save memory
    trajectoryh = nothing
    trajectoryl = nothing

    dI_da = (Imageh - Imagel) / (2 * h)  # Finite difference approximation

    #calculate the image at the central value
    Xcam = MVec4(camera_position(ro, th, phi, bhspin, Rout))
    trajectory = CalculateGeodesics(Xcam, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, bhspin, Rout, Rstop);
    Imagec = IpoleGeoIntensityIntegration(trajectory, freq, pixels_x, pixels_y, scale_factor, bhspin);
    trajectory = nothing

    return dI_da, Imagec
end

function armijo_line_search!(cost_func, compute_gradients!, x, grad, direction, bounds, 
                             args...; α=1e-4, β=0.5, initial_step=1.0, max_steps=10)
    """
    Improved Armijo line search with proper bounds handling and more aggressive stepping
    """
    step_size = initial_step
    x_new = similar(x)
    
    # Initial cost and directional derivative
    f0 = cost_func(x, args...)
    df0 = dot(grad, direction)
    println("grad = $grad, direction = $direction")
    println("  Line search: f0=$f0, df0=$df0, initial_step=$initial_step")

    #Tell if the direction is increasing or decreasing
    if direction[1] > 0
        println("  Direction for θo is increasing, testing higher θo values")
    else
        println("  Direction for θo is decreasing, testing lower θo values")
    end
    
    # If direction is not a descent direction, return current point
    if df0 >= 0
        @warn "Not a descent direction, df0 = $df0"
        #return x, f0, 0.0
    end
    
    best_x = copy(x)
    best_f = f0
    best_step = 0.0
    
    for i in 1:max_steps
        # Propose new point
        x_new .= x .+ step_size .* direction
        
        # Project onto bounds
        any_bounded = false
        for j in eachindex(x_new)
            old_val = x_new[j]
            x_new[j] = clamp(x_new[j], bounds[j][1], bounds[j][2])
            if x_new[j] != old_val
                any_bounded = true
            end
        end
        
        if any_bounded
            println("  Step $i: \e[31mHit bounds\e[0m")
            if x_new[2] == bounds[2][1] || x_new[2] == bounds[2][2]
                println(" Wanted to try Rhigh = $((x[2] .+ step_size .* direction[2] )* 120.0), but hit bounds at $(x_new[2] * 120.0)")
                step_size = (x_new[2] - x[2]) / direction[2]
                println("")
            end
            if x_new[1] == bounds[1][1] || x_new[1] == bounds[1][2]
                println(" Wanted to try θo = $((x[1] .+ step_size .* direction[1] )* 60), but hit bounds at $(x_new[1] * 60)")
                step_size = (x_new[1] - x[1]) / direction[1]
            end
            #continue
        end
        
        # Check if step is too small. Sometimes we get stuck in a loop of tiny steps
        step_norm = norm(x_new - x)
        if step_norm < 1e-20
            println(" \e[31mStep $i: Step norm too small ($step_norm), breaking\e[0m")
            println(" x_new = $x_new, x = $x, step_size = $step_size")
            break
        end
        
        # Evaluate cost at new point
        f_new = cost_func(x_new, args...)
        println("  Step $i: step_size=$step_size, f_new=$f_new, improvement=$(f0-f_new), Rhigh tested = $(x_new[2] * 120.0), θo tested = $(x_new[1] * 60)")
        
        # Keep track of best point found so far
        if f_new < best_f
            best_x .= x_new
            best_f = f_new
            best_step = step_size
        end
        
        # Check Armijo condition. If necessary, change alpha and beta to be more or less strict
        #armijo_threshold = f0 + α * step_size * df0
        armijo_threshold = f0 + α * step_size * df0
        if f_new <= armijo_threshold
            println("  \e[32mArmijo condition satisfied!\e[0m")
            println(" New cost function value: $f_new")
            return x_new, f_new, step_size, true
        end
        
        println("Reducing step size by beta factor: $β, Armijo Condition: $armijo_threshold, $f0, step_size=$step_size, df0=$df0")
        step_size *= β
        
        # If step becomes too small, return best point found
        if abs(x_new[1] - x[1]) < 1e-3 && abs(x_new[2] - x[2]) < 1e-3
            println("  \e[31mBoth parameter changes below 1e-3 threshold, returning best point found\e[0m")
            return best_x, best_f, best_step , false
        end
    end
    
    # If no Armijo condition met but we found improvements, return best
    if best_f < f0
        println("\e[31m  No Armijo satisfaction, but found improvement: $(f0 - best_f)\e[0m")
        return best_x, best_f, best_step, false
    end
    
    # If no improvement found at all, return original point
    println("\e[31mLine search failed to find any improvement\e[0m")
    return x, f0, best_step, false
end

# function armijo_line_search!(cost_func, x, grad, direction, bounds; 
#                              α=1e-4, β=0.5, initial_step=1.0, max_steps=15, 
#                              scales=ones(length(x))) # Optional: Pass [60.0, 2500.0] for nice printing
#     """
#     Projected Armijo Line Search.
#     Finds a step size that satisfies the sufficient decrease condition while respecting bounds.
#     """
    
#     # 1. Initial Checks
#     f0 = cost_func(x)
#     df0 = dot(grad, direction)
    
#     if df0 >= 0
#         @warn "Not a descent direction (df0 = $df0 >= 0). Line search aborted."
#         return x, f0, 0.0
#     end

#     # 2. Calculate Distance to Bounds (Look-ahead)
#     # This prevents checking points that are miles outside the bounds
#     max_feasible_step = Inf
#     for i in eachindex(x)
#         if direction[i] > 1e-9 # Moving towards upper bound
#             dist = bounds[i][2] - x[i]
#             max_feasible_step = min(max_feasible_step, dist / direction[i])
#         elseif direction[i] < -1e-9 # Moving towards lower bound
#             dist = bounds[i][1] - x[i]
#             max_feasible_step = min(max_feasible_step, dist / direction[i])
#         end
#     end
    
#     # Clamp initial step to the boundary distance
#     step_size = min(initial_step, max_feasible_step)
    
#     # Track best found in case we fail to satisfy strict Armijo but find a better point
#     best_x = copy(x)
#     best_f = f0
#     best_step = 0.0
#     found_better = false

#     println("  Line Search: f0 = $f0, Initial Step = $step_size (Max Feasible = $max_feasible_step)")

#     # 3. Backtracking Loop
#     for i in 1:max_steps
#         # A. Propose new point
#         x_new = x .+ step_size .* direction
        
#         # B. Safety Clamp (Just to handle floating point overshoot)
#         for k in eachindex(x_new)
#             x_new[k] = clamp(x_new[k], bounds[k][1], bounds[k][2])
#         end
        
#         # C. Evaluate
#         f_new = cost_func(x_new)
        
#         # Pretty print current physical parameters being tested
#         params_str = join([round(x_new[k] * scales[k], digits=4) for k in 1:length(x_new)], ", ")
#         println("   Step $i: size=$step_size, f_new=$f_new, Params=($params_str)")

#         # D. Check Sufficient Decrease (Armijo Condition)
#         # f(x + ad) <= f(x) + c1 * a * grad*d
#         target_f = f0 + α * step_size * df0
        
#         if f_new <= target_f
#             println("Armijo condition satisfied!")
#             return x_new, f_new, step_size
#         end

#         # E. Keep track if we found *any* improvement, even if not satisfying Armijo
#         if f_new < best_f
#             best_f = f_new
#             best_x .= x_new
#             best_step = step_size
#             found_better = true
#         end

#         # F. Reduce step size
#         step_size *= β
        
#         # G. Termination on tiny steps
#         if step_size < 1e-9
#             println("Step size too small. Stopping.")
#             break
#         end
#     end

#     # 4. Fallback
#     if found_better
#         println("Armijo not strictly satisfied, but taking best found improvement.")
#         return best_x, best_f, best_step
#     else
#         println("Line search failed. Staying at x.")
#         return x, f0, 0.0
#     end
# end

function true_conjugate_gradient_optimization(Iobs, ro, θoi, ai, freq, nx, ny, nmaxstep, 
                                            fovx, fovy, Rout, Rstop, σ_pixels=0.0; 
                                            cost_tol=2e-11, param_tol=1e-8, grad_tol=1e-10,
                                            max_iterations=200, cg_restart_freq=20,
                                            optimize_param::Symbol=:both)
    """
    True conjugate gradient optimization with proper scaling and convergence criteria
    
    Parameters:
    @Iobs: Observed image intensities (2D array) - This is the one we want to fit
    @ro: Observer radial coordinate
    @θoi: Initial guess for inclination angle in degrees
    @ai: Initial guess for spin parameter (0 to 0.994)
    @freq: Observed frequency, usually 230 GHz
    @nx, @ny: Image dimensions. The resolution of the image
    @nmaxstep: Maximum number of integration steps for geodesics
    @fovx, @fovy: Field of view in radians
    @Rout: Outer radius for geodesic integration
    @Rstop: Stopping radius for geodesic integration
    @σ_pixels: Standard deviation for Gaussian filter applied to intensities and gradients
    @cost_tol: Tolerance for cost function convergence
    @param_tol: Tolerance for parameter change convergence
    @grad_tol: Tolerance for gradient norm convergence
    @max_iterations: Maximum number of optimization iterations
    @cg_restart_freq: Frequency of conjugate gradient restarts
    @optimize_param: Symbol indicating which parameter(s) to optimize (:both, :theta, :spin)

    """
    # Pre-allocate trajectory arrays for each thread (this is going to be used in the autodiff function)
    num_threads = Threads.nthreads()
    thread_trajs = Vector{Vector{OfTraj}}(undef, num_threads)
    for tid in 1:num_threads
        thread_trajs[tid] = Vector{OfTraj}()
        sizehint!(thread_trajs[tid], nmaxstep)
    end

    # You can optimize either both parameters, only θo, or only a, it depends on the optimize_param argument. Make sure it is one of the three options.
    if !(optimize_param in [:both, :theta, :spin])
        throw(ArgumentError("optimize_param must be :both, :theta, or :spin"))
    end
    
    # Parameter scaling factors (normalize to similar ranges)
    θo_scale = 60.0
    a_scale = 0.6
    scales = [θo_scale, a_scale]
    
    # Initialize with scaled parameters
    x_scaled = [θoi / θo_scale, ai / a_scale]
    bounds_scaled = [(0.1 / θo_scale, 90.0 / θo_scale), 
                     (0.0 / a_scale, 0.994 / a_scale)]
    
    # Determine which parameters to optimize
    optimize_theta = optimize_param in [:both, :theta]
    optimize_spin = optimize_param in [:both, :spin]
    
    println("Optimization mode: $optimize_param")
    println("Optimizing θo: $optimize_theta, Optimizing a: $optimize_spin")
    
    dI_dθo = Matrix{Float64}(undef, nx, ny)
    dI_da = Matrix{Float64}(undef, nx, ny)
    I_calc = Matrix{Float64}(undef, nx, ny)
    
    function compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0)
        # Convert back to physical parameters
        θo_val = x_scaled_val[1] * θo_scale
        a_val = x_scaled_val[2] * a_scale
        println("Running AutoDiffGeoTrajEulerMethod with θo = $θo_val, a = $a_val and applying σ_pixels = $σ_pixels filter")
        # Compute intensities and derivatives
        Threads.@threads for i in 0:(nx-1)
            for j in 0:(ny-1)
                tid = Threads.threadid()
                dI_dθo_out = Ref{Float64}()
                intensity_out = Ref{Float64}()
                dI_da_out = Ref{Float64}()
                
                AutoDiffGeoTrajEulerMethod!(thread_trajs[tid],dI_dθo_out, intensity_out, dI_da_out,
                    ro, θo_val, a_val, nx, ny, nmaxstep, i, j, freq, fovx, fovy, Rout, Rstop)
                
                I_calc[i+1, j+1] = intensity_out[]
                dI_da[i+1, j+1] = dI_da_out[]
                dI_dθo[i+1, j+1] = dI_dθo_out[]
            end
        end
        I_calc = imfilter(I_calc, Kernel.gaussian(σ_pixels))
        dI_dθo = imfilter(dI_dθo, Kernel.gaussian(σ_pixels))
        dI_da = imfilter(dI_da, Kernel.gaussian(σ_pixels))
        cost = cost_func(Iobs, I_calc)
        grad_θo, grad_a = GradientofCostFunction(Iobs, I_calc, dI_dθo, dI_da)
        
        grad_scaled = [grad_θo * θo_scale, grad_a * a_scale]
        
        # Zero out gradients if evolving only theta or only spin. 
        if !optimize_theta
            grad_scaled[1] = 0.0
        end
        if !optimize_spin
            grad_scaled[2] = 0.0
        end
        
        return cost, grad_scaled
    end
    
    function constrained_armijo_line_search!(cost_func, compute_gradients!, x, grad, direction, bounds, args...; kwargs...)
        constrained_direction = copy(direction)
        if !optimize_theta
            constrained_direction[1] = 0.0
        end
        if !optimize_spin
            constrained_direction[2] = 0.0
        end
        
        return armijo_line_search!(cost_func, compute_gradients!, x, grad, constrained_direction, bounds, args...; kwargs...)
    end
        
    function check_convergence(cost, grad, cost_history, iteration)
        cost_converged = cost < cost_tol
        
        grad_norm = 0.0
        if optimize_theta
            grad_norm += grad[1]^2
        end
        if optimize_spin
            grad_norm += grad[2]^2
        end
        grad_norm = sqrt(grad_norm)
        grad_converged = false # For now, I'm not using grad convergence, maybe implement this later

        # Relative cost improvement over recent iterations
        rel_improvement_converged = false
        if iteration > 10 && length(cost_history) > 10
            recent_improvement = (cost_history[end-9] - cost) / abs(cost_history[end-9])
            rel_improvement_converged = recent_improvement < param_tol
        end
        
        println("  Convergence check: cost=$cost, grad_norm=$grad_norm")
        if iteration > 10
            recent_improvement = length(cost_history) > 10 ? (cost_history[end-9] - cost) / abs(cost_history[end-9]) : Inf
            println("  Recent relative improvement: $recent_improvement")
        end
        println("  Cost converged: $cost_converged, Grad converged: $grad_converged, Stagnant: $rel_improvement_converged")
        
        return cost_converged || grad_converged || rel_improvement_converged
    end

    last_x_computed = nothing
    last_cost = nothing
    last_grad = nothing

    function cached_compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0)
        if last_x_computed !== nothing && last_x_computed ≈ x_scaled_val
            println("Using cached computation for x = $x_scaled_val")
            return last_cost, last_grad
        end
        
        cost, grad = compute_cost_and_gradients(x_scaled_val, σ_pixels)
        last_x_computed = copy(x_scaled_val)
        last_cost = cost
        last_grad = copy(grad)
        return cost, grad
    end
    

    # Initial evaluation
    cost, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels)
    initial_cost = cost

    if check_convergence(cost, grad, [cost], 0)
        θo_final = x_scaled[1] * θo_scale
        a_final = x_scaled[2] * a_scale
        println("Initial solution already satisfies tolerance")
        return θo_final, a_final, [cost], 1
    end
    # Initialize CG variables
    direction = -copy(grad)
    # Zero out direction for fixed parameters
    if !optimize_theta
        direction[1] = 0.0
    end
    if !optimize_spin
        direction[2] = 0.0
    end
    
    costs = Float64[]
    push!(costs, cost)
    θos = Float64[]
    push!(θos, x_scaled[1] * θo_scale)
    as = Float64[]
    push!(as, x_scaled[2] * a_scale)
    θo_phys = x_scaled[1] * θo_scale
    a_phys = x_scaled[2] * a_scale

    println("Initial cost: $cost, Initial θo: $θo_phys, Initial a: $a_phys")
    println("Initial gradient norm: $(norm(grad))")
    
    x_old = copy(x_scaled)
    step_size = 0.0
    aggressive_initial_step = 0.0
    for iteration in 1:max_iterations
        println("\n--- Iteration $iteration ---")
        
        # Line search with constrained direction
        if (iteration == 1)
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
        else
            if(step_size < 0.05/(a_scale * direction[2] ) && optimize_spin)
                step_size = 0.05/(a_scale * direction[2] )
                println("Step size for a is too small, resetting to $step_size")
            end

            if(step_size < 3.0/(θo_scale * direction[1] ) && optimize_theta)
                step_size = 3.0/(θo_scale * direction[1] )
                println("Step size for θo is too small, resetting to $step_size")
            end

            aggressive_initial_step = step_size
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))

        end

        println("Trying aggressive initial step: $aggressive_initial_step, set at iteration $iteration")
        println("Cost before line search: $cost")
        cost_comparison = copy(cost)
        x_new, cost_new, step_size, sucess = constrained_armijo_line_search!(
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[1],
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[2],
            x_scaled, grad, direction, bounds_scaled,
            α=1e-5, β=0.5, initial_step=aggressive_initial_step, max_steps=15
        )
        
        # Check for sufficient decrease
        println("Line search completed: new cost = $cost_new, step size = $step_size")
        println("Initial cost after line search: $cost")
        absolute_improvement = cost_comparison - cost_new
        relative_improvement = absolute_improvement / max(abs(cost_comparison), 1e-16)
        println("Cost improvement: $absolute_improvement (relative: $relative_improvement)")
        
        if absolute_improvement <= 1e-16 && iteration > 1
            println("No significant improvement in line search, trying steepest descent")
            direction .= -grad
            # Zero out direction for fixed parameters
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_spin
                direction[2] = 0.0
            end
            
            x_new, cost_new, step_size, sucess = constrained_armijo_line_search!(
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[1],
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[2],
                x_scaled, grad, direction, bounds_scaled,
                α=1e-5, β=0.5, initial_step=step_size * 10
            )
            
            absolute_improvement = cost - cost_new
            if absolute_improvement <= 1e-16
                @warn "No improvement possible, stopping optimization"
                break
            end
        end
        
        if optimize_theta
            x_scaled[1] = x_new[1]
        end
        if optimize_spin
            x_scaled[2] = x_new[2]
        end
        
        param_change = norm(x_new - x_old)
        x_old .= x_scaled
        cost = copy(cost_new)
        push!(costs, cost)
        
        θo_phys = x_scaled[1] * θo_scale
        a_phys = x_scaled[2] * a_scale
        push!(θos, θo_phys)
        push!(as, a_phys)
        println("Updated: cost = $cost, θo = $θo_phys, a = $a_phys, step = $step_size")
        println("Parameter change magnitude: $param_change")
        
        converged = check_convergence(cost, grad, costs, iteration)
        
        relative_param_change = param_change / max(norm(x_scaled), 1e-10)
        if relative_param_change < param_tol
            println("Relative parameter change too small ($relative_param_change), may have converged")
            converged = true
        end
        if converged
            println("Converged! Final θo = $θo_phys, Final a = $a_phys")
            return θos, as, costs, max_iterations
        end
        
        grad_old = copy(grad)
        _, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels)
        
        if iteration % cg_restart_freq == 0 || norm(grad) > 10 * norm(grad_old)
            direction .= -grad
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_spin
                direction[2] = 0.0
            end
            println("CG restart at iteration $iteration")
        else
            # Polak-Ribière formula
            grad_diff = grad - grad_old
            beta_denom = dot(grad_old, grad_old)
            
            if beta_denom > 1e-16
                beta = max(0.0, dot(grad, grad_diff) / beta_denom)
                
                beta = min(beta, 2.0)
                
                direction .= -grad .+ beta .* direction
                
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_spin
                    direction[2] = 0.0
                end
                
                active_grad_norm = 0.0
                active_dir_norm = 0.0
                active_dot = 0.0
                
                if optimize_theta
                    active_grad_norm += grad[1]^2
                    active_dir_norm += direction[1]^2
                    active_dot += direction[1] * grad[1]
                end
                if optimize_spin
                    active_grad_norm += grad[2]^2
                    active_dir_norm += direction[2]^2
                    active_dot += direction[2] * grad[2]
                end
                
                active_grad_norm = sqrt(active_grad_norm)
                active_dir_norm = sqrt(active_dir_norm)
                
                if active_dot >= -1e-10 * active_dir_norm * active_grad_norm
                    direction .= -grad
                    if !optimize_theta
                        direction[1] = 0.0
                    end
                    if !optimize_spin
                        direction[2] = 0.0
                    end
                    println("Reset to steepest descent (not descent direction)")
                end
                
                println("CG update: beta = $beta")
            else
                direction .= -grad
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_spin
                    direction[2] = 0.0
                end
                println("Beta = $beta, Reset to steepest descent (small denominator)")
            end
        end
    end
    
    @warn "Maximum iterations reached without convergence"
    θo_final = x_scaled[1] * θo_scale
    a_final = x_scaled[2] * a_scale
    return θos, as, costs, max_iterations
end



function true_conjugate_gradient_optimization_GRMHD(Iobs, ro, θoi, Rhighi, freq, nx, ny, nmaxstep, 
                                            fovx, fovy, Rout, Rstop, σ_pixels=0.0; 
                                            cost_tol=2e-11, param_tol=1e-8, grad_tol=1e-10,
                                            max_iterations=200, cg_restart_freq=20,
                                            optimize_param::Symbol=:both, dump_filepath = nothing, sensemode = "AD")
    """
    True conjugate gradient optimization with proper scaling and convergence criteria
    
    Parameters:
    @Iobs: Observed image intensities (2D array) - This is the one we want to fit
    @ro: Observer radial coordinate
    @θoi: Initial guess for inclination angle in degrees
    @ai: Initial guess for spin parameter (0 to 0.994)
    @freq: Observed frequency, usually 230 GHz
    @nx, @ny: Image dimensions. The resolution of the image
    @nmaxstep: Maximum number of integration steps for geodesics
    @fovx, @fovy: Field of view in radians
    @Rout: Outer radius for geodesic integration
    @Rstop: Stopping radius for geodesic integration
    @σ_pixels: Standard deviation for Gaussian filter applied to intensities and gradients
    @cost_tol: Tolerance for cost function convergence
    @param_tol: Tolerance for parameter change convergence
    @grad_tol: Tolerance for gradient norm convergence
    @max_iterations: Maximum number of optimization iterations
    @cg_restart_freq: Frequency of conjugate gradient restarts
    @optimize_param: Symbol indicating which parameter(s) to optimize (:both, :theta, :spin)

    """
    # Pre-allocate trajectory arrays for each thread (this is going to be used in the autodiff function)
    num_threads = Threads.nthreads()
    simulation_data = load_data(dump_filepath, Rhighi);

    thread_trajs = Vector{Vector{OfTraj}}(undef, num_threads+1)
    for tid in 1:(num_threads+1)
        thread_trajs[tid] = Vector{OfTraj}()
        sizehint!(thread_trajs[tid], nmaxstep)
    end

    # You can optimize either both parameters, only θo, or only a, it depends on the optimize_param argument. Make sure it is one of the three options.
    if !(optimize_param in [:both, :theta, :Rhigh])
        throw(ArgumentError("optimize_param must be :both, :theta, or :Rhigh"))
    end

    if !(sensemode in ["AD", "FD"])
        throw(ArgumentError("sensemode arg must be either 'AD' or 'FD'"))
    end

    sucess::Bool = false
    
    # Parameter scaling factors (normalize to similar ranges)
    θo_scale = 60.0
    Rhigh_scale = 2500.0
    scales = [θo_scale, Rhigh_scale]
    
    # Initialize with scaled parameters
    x_scaled = [θoi / θo_scale, Rhighi / Rhigh_scale]
    bounds_scaled = [(0.1 / θo_scale, 179.9 / θo_scale), 
                     (0.0 / Rhigh_scale, 200.0 / Rhigh_scale)]
    
    # Determine which parameters to optimize
    optimize_theta = optimize_param in [:both, :theta]
    optimize_Rhigh = optimize_param in [:both, :Rhigh]
    
    println("Optimization mode: $optimize_param")
    println("Optimizing θo: $optimize_theta, Optimizing Rhigh: $optimize_Rhigh")
    
    dI_dθo = Matrix{Float64}(undef, nx, ny)
    dI_dRhigh = Matrix{Float64}(undef, nx, ny)
    I_calc = Matrix{Float64}(undef, nx, ny)
    
    function compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0, simulation_data = nothing, sensemode = "AD")
        # Convert back to physical parameters
        θo_val = x_scaled_val[1] * θo_scale
        Rhigh_val = x_scaled_val[2] * Rhigh_scale
        if optimize_Rhigh
            simulation_data = load_data(dump_filepath, Rhigh_val);
        end
        println("\n Running AutoDiffGeoTrajEulerMethod with θo = \e[32m$θo_val\e[0m, Rhigh = \e[32m$Rhigh_val\e[0m and applying σ_pixels = \e[32m$σ_pixels\e[0m filter")
        if(sensemode == "AD")
            Threads.@threads for i in 0:(nx-1)
                for j in 0:(ny-1)
                    tid = Threads.threadid()
                    dI_dθo_out = Ref{Float64}()
                    intensity_out = Ref{Float64}()
                    dI_dRhigh_out = Ref{Float64}()
                    
                    AutoDiffGeoTrajEulerMethod_GRMHD!(thread_trajs[tid], dI_dθo_out, intensity_out, dI_dRhigh_out,
                        ro, θo_val, phi, params.a, nx, ny, nmaxstep, i, j, freq, fovx, fovy, Rout, Rstop, simulation_data)
                    I_calc[i+1, j+1] = intensity_out[]
                    dI_dRhigh[i+1, j+1] = dI_dRhigh_out[]
                    dI_dθo[i+1, j+1] = dI_dθo_out[]
                end
            end
        elseif(sensemode == "FD")
            println("Using Finite Differences to compute gradients")
            dI_dθo, I_calc = FiniteDifferencesθ(ro, θo_val, phi, DXsize, DYsize, nx, ny, SourceD, freq, nmaxstep, 1e-4, params.a, Rout, Rstop, simulation_data)
            if(optimize_Rhigh)
                error("Finite Differences for Rhigh not implemented yet")
                dI_dRhigh, _ = FiniteDifferences_Rhigh(ro, θo_val, phi, DXsize, DYsize, nx, ny, SourceD, freq, nmaxstep, 1e-4, params.a, Rout, Rstop, simulation_data)
            end
        else
            throw(ArgumentError("sensemode must be either 'AD' or 'FD'"))
        end
        I_calc = imfilter(I_calc, Kernel.gaussian(σ_pixels))
        dI_dθo = imfilter(dI_dθo, Kernel.gaussian(σ_pixels))
        dI_dRhigh = imfilter(dI_dRhigh, Kernel.gaussian(σ_pixels))
        cost = cost_func(Iobs, I_calc)
        grad_θo, grad_Rhigh = GradientofCostFunction(Iobs, I_calc, dI_dθo, dI_dRhigh)
        
        grad_scaled = [grad_θo * θo_scale, grad_Rhigh * Rhigh_scale]
        
        # Zero out gradients if evolving only theta or only rhigh. 
        if !optimize_theta
            grad_scaled[1] = 0.0
        end
        if !optimize_Rhigh
            grad_scaled[2] = 0.0
        end
        
        return cost, grad_scaled
    end
    
    function constrained_armijo_line_search!(cost_func, compute_gradients!, x, grad, direction, bounds, args...; kwargs...)
        constrained_direction = copy(direction)
        if !optimize_theta
            constrained_direction[1] = 0.0
        end
        if !optimize_Rhigh
            constrained_direction[2] = 0.0
        end
        
        return armijo_line_search!(cost_func, compute_gradients!, x, grad, constrained_direction, bounds, args...; kwargs...)
    end
        
    function check_convergence(cost, grad, cost_history, iteration)
        cost_converged = cost < cost_tol
        
        grad_norm = 0.0
        if optimize_theta
            grad_norm += grad[1]^2
        end
        if optimize_Rhigh
            grad_norm += grad[2]^2
        end
        grad_norm = sqrt(grad_norm)
        grad_converged = false # For now, I'm not using grad convergence, maybe implement this later

        # Relative cost improvement over recent iterations
        rel_improvement_converged = false
        if iteration > 10 && length(cost_history) > 10
            recent_improvement = (cost_history[end-9] - cost) / abs(cost_history[end-9])
            rel_improvement_converged = recent_improvement < param_tol
        end
        
        println("  Convergence check: cost=$cost, grad_norm=$grad_norm")
        if iteration > 10
            recent_improvement = length(cost_history) > 10 ? (cost_history[end-9] - cost) / abs(cost_history[end-9]) : Inf
            println("  Recent relative improvement: $recent_improvement")
        end
        println("  Cost converged: $cost_converged, Grad converged: $grad_converged, Stagnant: $rel_improvement_converged")
        
        return cost_converged || grad_converged || rel_improvement_converged
    end

    last_x_computed = nothing
    last_cost = nothing
    last_grad = nothing

    function cached_compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0, simulation_data = nothing, sensemode = "AD")
        if last_x_computed !== nothing && last_x_computed ≈ x_scaled_val
            println("Using cached computation for x = $x_scaled_val")
            return last_cost, last_grad
        end
        
        cost, grad = compute_cost_and_gradients(x_scaled_val, σ_pixels, simulation_data, sensemode)
        last_x_computed = copy(x_scaled_val)
        last_cost = cost
        last_grad = copy(grad)
        return cost, grad
    end
    

    # Initial evaluation
    cost, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels, simulation_data, sensemode)
    initial_cost = cost

    if check_convergence(cost, grad, [cost], 0)
        θo_final = x_scaled[1] * θo_scale
        Rhigh_final = x_scaled[2] * Rhigh_scale
        println("Initial solution already satisfies tolerance")
        return θo_final, Rhigh_final, [cost], 1
    end
    # Initialize CG variables
    direction = -copy(grad)
    # Zero out direction for fixed parameters
    if !optimize_theta
        direction[1] = 0.0
    end
    if !optimize_Rhigh
        direction[2] = 0.0
    end
    
    costs = Float64[]
    push!(costs, cost)
    θos = Float64[]
    push!(θos, x_scaled[1] * θo_scale)
    Rhighs = Float64[]
    push!(Rhighs, x_scaled[2] * Rhigh_scale)
    θo_phys = x_scaled[1] * θo_scale
    Rhigh_phys = x_scaled[2] * Rhigh_scale

    println("Initial cost: $cost, Initial θo: $θo_phys, Initial Rhigh: $Rhigh_phys")
    println("Initial gradient norm: $(norm(grad))")
    
    x_old = copy(x_scaled)
    step_size = 0.0
    aggressive_initial_step = 0.0
    for iteration in 1:max_iterations
        println("\n--- Iteration $iteration ---")
        
        # Line search with constrained direction
        if (iteration == 1)
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
            #aggressive_initial_step = 0.3/norm(grad)
        else
            if(step_size < 0.05/(Rhigh_scale * direction[2] ) && optimize_Rhigh)
                step_size = 0.05/(Rhigh_scale * direction[2] )
                println("Step size for Rhigh is too small, resetting to $step_size")
            end

            if(step_size < 0.1/(θo_scale * direction[1] ) && optimize_theta)
                step_size = 0.1/(θo_scale * direction[1] )
                println("Step size for θo is too small, resetting to $step_size")
            end
            if(sucess)
                println("Line search successful, using step size of $step_size for next iteration")
                aggressive_initial_step = step_size
            else
                println("Line search failed to find a better point, using aggressive initial step for next iteration")
                aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
            end
        end

        cost_comparison = copy(cost)
        x_new, cost_new, step_size, sucess = constrained_armijo_line_search!(
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels, simulation_data, sensemode)[1],
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels, simulation_data, sensemode)[2],
            x_scaled, grad, direction, bounds_scaled,
            α=1e-5, β=0.5, initial_step=aggressive_initial_step, max_steps=15
        )
        
        # Check for sufficient decrease
        absolute_improvement = cost_comparison - cost_new
        relative_improvement = absolute_improvement / max(abs(cost_comparison), 1e-16)
        println("\e[34mCost improvement: $absolute_improvement (relative: $relative_improvement)\e[0m")
        
        if absolute_improvement <= 1e-16 && iteration > 1
            println("\e[31mNo significant improvement in line search, trying steepest descent\e[0m")
            direction .= -grad
            # Zero out direction for fixed parameters
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_Rhigh
                direction[2] = 0.0
            end
            
            x_new, cost_new, step_size, sucess = constrained_armijo_line_search!(
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels, simulation_data, sensemode)[1],
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels, simulation_data, sensemode)[2],
                x_scaled, grad, direction, bounds_scaled,
                α=1e-5, β=0.5, initial_step=0.01
            )
            
            absolute_improvement = cost - cost_new
            if absolute_improvement <= 1e-16
                @warn "No improvement possible, stopping optimization"
                break
            end
        end
        
        if optimize_theta
            x_scaled[1] = x_new[1]
        end
        if optimize_Rhigh
            x_scaled[2] = x_new[2]
        end
        
        param_change = norm(x_new - x_old)
        x_old .= x_scaled
        cost = copy(cost_new)
        push!(costs, cost)
        
        θo_phys = x_scaled[1] * θo_scale
        Rhigh_phys = x_scaled[2] * Rhigh_scale
        push!(θos, θo_phys)
        push!(Rhighs, Rhigh_phys)
        println("Updated: cost = $cost, θo = $θo_phys, Rhigh = $Rhigh_phys, step = $step_size")
        println("Parameter change magnitude: $param_change")
        
        converged = check_convergence(cost, grad, costs, iteration)
        
        relative_param_change = param_change / max(norm(x_scaled), 1e-10)
        if relative_param_change < param_tol
            println("Relative parameter change too small ($relative_param_change), may have converged")
            println("However, we won't stop")

            converged = false
        end
        if converged
            println("Converged! Final θo = $θo_phys, Final Rhigh = $Rhigh_phys")
            return θos, Rhighs, costs, max_iterations
        end
        
        grad_old = copy(grad)
        _, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels, simulation_data, sensemode)
        
        if iteration % cg_restart_freq == 0 || norm(grad) > 10 * norm(grad_old)
            direction .= -grad
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_spin
                direction[2] = 0.0
            end
            println("CG restart at iteration $iteration")
        else
            # Polak-Ribière formula
            grad_diff = grad - grad_old
            beta_denom = dot(grad_old, grad_old)
            
            if beta_denom > 1e-16
                beta = max(0.0, dot(grad, grad_diff) / beta_denom)
                
                beta = min(beta, 2.0)
                
                direction .= -grad .+ beta .* direction
                
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_Rhigh
                    direction[2] = 0.0
                end
                
                active_grad_norm = 0.0
                active_dir_norm = 0.0
                active_dot = 0.0
                
                if optimize_theta
                    active_grad_norm += grad[1]^2
                    active_dir_norm += direction[1]^2
                    active_dot += direction[1] * grad[1]
                end
                if optimize_Rhigh
                    active_grad_norm += grad[2]^2
                    active_dir_norm += direction[2]^2
                    active_dot += direction[2] * grad[2]
                end
                
                active_grad_norm = sqrt(active_grad_norm)
                active_dir_norm = sqrt(active_dir_norm)
                
                if active_dot >= -1e-10 * active_dir_norm * active_grad_norm
                    direction .= -grad
                    if !optimize_theta
                        direction[1] = 0.0
                    end
                    if !optimize_Rhigh
                        direction[2] = 0.0
                    end
                    println("Reset to steepest descent (not descent direction)")
                end
                
                println("CG update: beta = $beta")
            else
                direction .= -grad
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_Rhigh
                    direction[2] = 0.0
                end
                println("Reset to steepest descent (small denominator)")
            end
        end
    end
    
    @warn "Maximum iterations reached without convergence"
    θo_final = x_scaled[1] * θo_scale
    Rhigh_final = x_scaled[2] * Rhigh_scale
    return θos, Rhighs, costs, max_iterations
end

# function true_conjugate_gradient_optimization_GRMHD(Iobs, ro, θoi, Rhighi, freq, nx, ny, nmaxstep, 
#                                                     fovx, fovy, Rout, Rstop, σ_pixels=0.0; 
#                                                     # Optimization Hyperparameters
#                                                     cost_tol=2e-11, param_tol=1e-8,
#                                                     max_iterations=200, cg_restart_freq=20,
#                                                     optimize_param::Symbol=:both, dump_filepath=nothing, sensemode="AD")

#     # ========================================================================================
#     # 1. SETUP & INITIALIZATION
#     # ========================================================================================
    
#     # --- Scaling & Masking ---
#     # Scales ensure the optimization treats 1 degree of Theta ~ 1 unit of Rhigh
#     scales = [60.0, 2500.0]  # [θo, Rhigh]
#     x = [θoi, Rhighi] ./ scales
    
#     # Create a mask to handle parameter fixing (Math instead of If-Statements)
#     # Mask = [1, 1] for both, [1, 0] for theta only, [0, 1] for Rhigh only
#     mask = [1.0, 1.0] 
#     if optimize_param == :theta; mask[2] = 0.0; end
#     if optimize_param == :Rhigh; mask[1] = 0.0; end
    
#     # --- Memory Pre-allocation ---
#     # Pre-allocate thread-local trajectory buffers for the AutoDiff engine
#     thread_trajs = [sizehint!(Vector{OfTraj}(), nmaxstep) for _ in 1:Threads.nthreads()]
    
#     # History containers
#     history = (cost=Float64[], θ=Float64[], Rhigh=Float64[])

#     # ========================================================================================
#     # 2. OBJECTIVE FUNCTION ( The Physics Interface )
#     # ========================================================================================
#     # This function handles unscaling, data loading, simulation, and gradient calculation.
    
#     function compute_physics_obj_grad(x_scaled)
#         # A. Unscale parameters to physical units
#         θ_phys, Rhigh_phys = x_scaled .* scales

#         # B. Load Simulation Data (Only if necessary)
#         # If optimizing Theta only, Rhigh is constant, so we load data once outside (conceptually)
#         # If Rhigh varies, we must reload/interpolate data every step.
#         sim_data = (optimize_param == :theta) ? load_data(dump_filepath, Rhighi) : load_data(dump_filepath, Rhigh_phys)

#         # C. Run Simulation & Compute Gradients
#         # We pre-allocate output matrices to avoid GC overhead
#         dI_dθ = zeros(nx, ny)
#         dI_dR = zeros(nx, ny)
#         I_calc = zeros(nx, ny)

#         if sensemode == "AD"
#             Threads.@threads for i in 0:(nx-1)
#                 for j in 0:(ny-1)
#                     tid = Threads.threadid()
#                     # Refs for output
#                     dI_dθ_ref, I_ref, dI_dR_ref = Ref(0.0), Ref(0.0), Ref(0.0)
                    
#                     # Call the existing AutoDiff Engine
#                     AutoDiffGeoTrajEulerMethod_GRMHD!(thread_trajs[tid], dI_dθ_ref, I_ref, dI_dR_ref,
#                                                       ro, θ_phys, 0.0, 0.0, nx, ny, nmaxstep, i, j, freq, 
#                                                       fovx, fovy, Rout, Rstop, sim_data)
                                                      
#                     I_calc[i+1, j+1]   = I_ref[]
#                     dI_dθ[i+1, j+1]    = dI_dθ_ref[]
#                     dI_dR[i+1, j+1]    = dI_dR_ref[]
#                 end
#             end
#         else # FD Mode
#             # Call external Finite Difference wrapper
#             dI_dθ, I_calc = FiniteDifferencesθ(ro, θ_phys, 0.0, nx, ny, freq, nmaxstep, Rout, Rstop, sim_data) 
#             if mask[2] > 0.0
#                  dI_dR, _ = FiniteDifferences_Rhigh(ro, θ_phys, 0.0, nx, ny, freq, nmaxstep, Rout, Rstop, sim_data)
#             end
#         end

#         # D. Apply Smoothing (if σ > 0)
#         if σ_pixels > 0
#             k = Kernel.gaussian(σ_pixels)
#             I_calc  = imfilter(I_calc, k)
#             dI_dθ   = imfilter(dI_dθ, k)
#             dI_dR   = imfilter(dI_dR, k)
#         end

#         # E. Calculate Cost & Scalar Gradients
#         cost = cost_func(Iobs, I_calc)
#         g_θ, g_R = GradientofCostFunction(Iobs, I_calc, dI_dθ, dI_dR)

#         # F. Scale & Mask Gradients
#         # Chain rule: dCost/dx_scaled = dCost/dPhys * dPhys/dx_scaled
#         grad_scaled = [g_θ, g_R] .* scales 
#         grad_scaled .*= mask # Zero out gradients for fixed parameters

#         return cost, grad_scaled
#     end

#     # --- Memoization Wrapper ---
#     # Prevents re-computing physics if the line search queries the same point twice.
#     last_x, last_cost, last_grad = nothing, nothing, nothing
#     function cached_obj_grad(x_val)
#         if last_x !== nothing && last_x ≈ x_val
#             return last_cost, copy(last_grad)
#         end
#         c, g = compute_physics_obj_grad(x_val)
#         last_x, last_cost, last_grad = copy(x_val), c, copy(g)
#         return c, g
#     end

#     # ========================================================================================
#     # 3. OPTIMIZATION LOOP ( The Math )
#     # ========================================================================================
    
#     # Initial Eval
#     cost, grad = cached_obj_grad(x)
#     direction = -grad # Initial direction is Steepest Descent
    
#     # Save initial state
#     push!(history.cost, cost); push!(history.θ, x[1]*scales[1]); push!(history.Rhigh, x[2]*scales[2])
#     println("Initial Cost: $cost | Gradient Norm: $(norm(grad))")

#     for iter in 1:max_iterations
#         println("\n--- Iteration $iter ---")
        
#         # A. Armijo Line Search
#         # Heuristic: aggressive step for first iter, otherwise 1.0 (Newton-like) or conservative
#         initial_step = (iter == 1) ? (0.3 / max(norm(grad), 1e-12)) : 1.0 
        
#         # Note: We pass the Masked direction. The bounds check happens inside line_search
#         # x_new, cost_new, alpha = armijo_line_search!(
#         #     v -> cached_obj_grad(v)[1], # Function that returns just cost
#         #     x, grad, direction;         # Current state
#         #     initial_step=initial_step,
#         #     bounds=[(0.1/scales[1], 179.9/scales[1]), (0.0, 200.0/scales[2])]
#         # )

#         # 2. Pass it as the 5th argument
#         x_new, cost_new, alpha = armijo_line_search!(
#             v -> cached_obj_grad(v)[1], 
#             x, 
#             grad, 
#             direction, 
#             [(0.1/scales[1], 179.9/scales[1]), (0.0, 200.0/scales[2])];
#             initial_step=initial_step,
#             scales=scales
#         )

#         # B. Convergence Check & Updates
#         param_diff = norm(x_new - x)
#         cost_diff  = cost - cost_new
        
#         x = x_new
#         push!(history.cost, cost_new)
#         push!(history.θ, x[1]*scales[1])
#         push!(history.Rhigh, x[2]*scales[2])

#         println("  Step size: $alpha | New Cost: $cost_new")
        
#         if cost_new < cost_tol || param_diff < param_tol || (cost_diff < 1e-16 && iter > 5)
#             println("✅ Converged at iteration $iter")
#             break
#         end
#         cost = cost_new

#         # C. Compute New Gradient (Fast, likely cached from line search)
#         grad_old = copy(grad)
#         _, grad = cached_obj_grad(x)

#         # D. Conjugate Gradient Update (Polak-Ribière)
#         beta = 0.0
        
#         # Restart logic: Every N steps OR if gradient orthogonality is lost (Powell's restart)
#         should_restart = (iter % cg_restart_freq == 0) || (abs(dot(grad, grad_old)) > 0.2 * dot(grad, grad))
        
#         if !should_restart
#             # Polak-Ribière Formula
#             denom = dot(grad_old, grad_old)
#             if denom > 1e-16
#                 beta = max(0.0, dot(grad, grad - grad_old) / denom)
#                 beta = min(beta, 2.0) # Safety cap
#             end
#         else
#             println("CG Restart")
#         end

#         # E. Update Direction
#         # d_{k+1} = -g_{k+1} + β * d_k
#         direction = -grad + beta * direction
        
#         # F. Descent Safety Check
#         # If the new direction is not a descent direction (angle with gradient < 90 deg), reset.
#         if dot(direction, grad) >= 0 
#             println("Direction not descending. Resetting to gradient.")
#             direction = -grad
#             beta = 0.0
#         end

#         # G. Apply Mask (Safety net to ensure fixed params stay fixed)
#         direction .*= mask 
#     end

#     return history.θ, history.Rhigh, history.cost, max_iterations
# end