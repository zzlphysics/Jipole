include("../src/main.jl")
using ProgressMeter
using DelimitedFiles

println("Available threads: ", Threads.nthreads())
println("RHO_unit:", RHO_unit)

# --- Configuration & Paths ---
dump_filepath = "../sample_dump_SANE_a+0.94_MKS_0900.h5";
const params = read_header(dump_filepath);
output_file = "grid_errors_theta_Rhigh.txt"

# --- Constants & Baseline (Truth) ---
const Rhigh_truth = 20.0 
const th_truth = 163.0   # Set your "Ground Truth" observer angle here
const trat_small = 1.0 
const beta_crit = 1.0 
const th_beg = 1.74e-2 
const sigma_cut = 1.0
const sigma_cut_high = -1.0;

const ro = 1000.0
const phi = 0.0
const pixels_x = 128
const pixels_y = 128
const SourceD = 16.9e6 * PC
const Rstop = 100.0
const Rh = 1 + sqrt(1. - params.a * params.a);
const freq = 230e9;
const DXsize = SourceD/L_unit/MUAS_PER_RAD * 160
const DYsize = SourceD/L_unit/MUAS_PER_RAD * 160
const fovx = DXsize/ro
const fovy = DYsize/ro
const xoff = 0.0
const yoff = 0.0
const maxnstep = 15000
const freq_unitless = freq * HPL/(ME * CL * CL) 

# --- Ranges for the Grid Search ---
# theta from 2 to 178 in steps of 2
# Rhigh from 1 to 100 in steps of 5
th_values = collect(1.0:2.0:179.0)
Rhigh_values = collect(1.0:5.0:100.0)

# ---------------------------------------------------------
# 1. GENERATE THE GROUND TRUTH IMAGE
# ---------------------------------------------------------
println("\n>>> Generating Ground Truth Image (θ=$th_truth, Rhigh=$Rhigh_truth)")
Xcamera_truth = MVec4(camera_position(ro, th_truth, phi, params.a, params.Rout))
sim_data_truth = load_data(dump_filepath, Rhigh_truth)
Image_truth = zeros(Float64, pixels_x, pixels_y)

p_truth = Progress(pixels_x * pixels_y; desc = "Raytracing Truth...")
@time begin
    Threads.@threads for i in 0:(pixels_x - 1)
        for j in 0:(pixels_y - 1)
            traj = Vector{OfTraj}()
            sizehint!(traj, maxnstep)
            nstep = get_pixel(traj, i, j, Xcamera_truth, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, params.a, Rh, params.Rout, Rstop, xoff, yoff) 
            resize!(traj, nstep)
            integrate_emission!(traj, nstep, Image_truth, i + 1, j + 1, freq, params.a, sim_data_truth)
            next!(p_truth)
        end
    end
    Image_truth *= freq^3
end

# ---------------------------------------------------------
# 2. GRID SEARCH LOOP
# ---------------------------------------------------------
# Initialize file
open(output_file, "w") do io
    println(io, "Theta\tRhigh\tNMSE")
end

println("\n>>> Starting Grid Search: $(length(th_values)) angles x $(length(Rhigh_values)) Rhigh values")

for th_now in th_values
    # Pre-calculate camera for this theta (saves time inside Rhigh loop)
    Xcamera_now = MVec4(camera_position(ro, th_now, phi, params.a, params.Rout))
    
    for R_now in Rhigh_values
        println("\n--- Testing: θ = $th_now, Rhigh = $R_now ---")
        
        sim_data_now = load_data(dump_filepath, R_now)
        Image_now = zeros(Float64, pixels_x, pixels_y)
        
        p = Progress(pixels_x * pixels_y; desc = "Raytracing...")
        @time begin
            Threads.@threads for i in 0:(pixels_x - 1)
                for j in 0:(pixels_y - 1)
                    traj = Vector{OfTraj}()
                    sizehint!(traj, maxnstep)
                    nstep = get_pixel(traj, i, j, Xcamera_now, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, params.a, Rh, params.Rout, Rstop, xoff, yoff) 
                    resize!(traj, nstep)
                    integrate_emission!(traj, nstep, Image_now, i + 1, j + 1, freq, params.a, sim_data_now)
                    next!(p)
                end
            end
            Image_now *= freq^3
        end
        
        # Calculate error
        err = cost_func(Image_truth, Image_now)
        
        # Stream results to file immediately
        open(output_file, "a") do io
            println(io, "$th_now\t$R_now\t$err")
        end
        println("Result: NMSE = $err (Saved)")
    end
end

println("\nGrid search complete. Data in $output_file")