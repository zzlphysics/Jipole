include("../src/main.jl")
using ProgressMeter
using DelimitedFiles

println("Available threads: ", Threads.nthreads())
println("RHO_unit:", RHO_unit)
dump_filepath = "../sample_dump_SANE_a+0.94_MKS_0900.h5";
const params = read_header(dump_filepath);

# Set the baseline Rhigh (trat_large) to generate our ground truth
const Rhigh_truth = 60.0 
const trat_small = 1.0 
const beta_crit = 1.0 
const th_beg = 1.74e-2 
const sigma_cut = 1.0
const sigma_cut_high = -1.0;

# Setting up the parameters
const ro = 1000.0
const th = 163.0
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

# ---------------------------------------------------------
# 1. GENERATE THE BASELINE (TRUTH) IMAGE
# ---------------------------------------------------------

# Find camera in native coordinates (only needs to be done once since angle is fixed)
const Xcamera = MVec4(camera_position(ro, th, phi, params.a, params.Rout))

scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)
println("scale_factor = $scale_factor")

# Load baseline simulation data
println("Loading baseline simulation data for Rhigh = $Rhigh_truth...")
const simulation_data_truth = load_data(dump_filepath, Rhigh_truth);

println("Utilizing $(Threads.nthreads()) threads for baseline geodesic calculation.")
p_truth = Progress(pixels_x * pixels_y; desc = "Raytracing Truth Image...", showspeed = true, barlen = 30)

const freq_unitless = freq * HPL/(ME * CL * CL) 
Image_truth = zeros(Float64, pixels_x, pixels_y)

@time begin
    Threads.@threads for i in 0:(pixels_x - 1)
        tid = Threads.threadid()
        for j in 0:(pixels_y - 1)
            traj = Vector{OfTraj}()
            sizehint!(traj, maxnstep)
            nstep = get_pixel(traj, i, j, Xcamera, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, params.a, Rh, params.Rout, Rstop, xoff, yoff) 
            
            resize!(traj, nstep)
            integrate_emission!(traj, nstep, Image_truth, i + 1, j + 1, freq, params.a, simulation_data_truth)
            
            ProgressMeter.next!(p_truth; showvalues = [(:thread_id, tid), (:pixel, "($i, $j)"), (:total_done, "$(i*pixels_y + j)/$(pixels_x * pixels_y)")])
        end
    end
    Image_truth *= freq^3;
end
finish!(p_truth);


# ---------------------------------------------------------
# 2. DEFINE THE EVALUATION FUNCTION
# ---------------------------------------------------------

function measure_error_Rhigh(Rhigh)
    # Load new physics data for the current Rhigh
    simulation_data = load_data(dump_filepath, Rhigh)

    Image = zeros(Float64, pixels_x, pixels_y)
    
    p = Progress(pixels_x * pixels_y; desc = "Raytracing Image (Rhigh=$Rhigh)...", showspeed = true, barlen = 30)

    @time begin
        Threads.@threads for i in 0:(pixels_x - 1)
            tid = Threads.threadid()
            for j in 0:(pixels_y - 1)
                traj = Vector{OfTraj}()
                sizehint!(traj, maxnstep)
                
                nstep = get_pixel(traj, i, j, Xcamera, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, params.a, Rh, params.Rout, Rstop, xoff, yoff)

                resize!(traj, nstep)
                integrate_emission!(traj, nstep, Image, i + 1, j + 1, freq, params.a, simulation_data)
                
                ProgressMeter.next!(p; showvalues = [(:thread_id, tid), (:pixel, "($i, $j)"), (:total_done, "$(i*pixels_y + j)/$(pixels_x * pixels_y)")])
            end
        end
        Image *= freq^3
    end
    finish!(p);

    return cost_func(Image_truth, Image)
end

# ---------------------------------------------------------
# 3. LOOP OVER RHIGH VALUES
# ---------------------------------------------------------

Rhigh_values = collect(1.0:1.0:100.0)
errs = zeros(Float64, length(Rhigh_values))

output_file = "Rhigh_errors_output_MKS_163_128_Rhigh60.txt"

# Create/overwrite the file with the header
open(output_file, "w") do io
    println(io, "Rhigh\tError")
end

println("\nStarting evaluation of $(length(Rhigh_values)) Rhigh values...")

for k in 1:length(Rhigh_values)
    Rhigh = Rhigh_values[k]

    println("\n========================================")
    println("Evaluating $k/$(length(Rhigh_values)): Rhigh = $Rhigh")
    println("========================================")

    err = measure_error_Rhigh(Rhigh)
    errs[k] = err

    println("-> Error for Rhigh = $Rhigh is: $err")

    # Save immediately to avoid data loss
    open(output_file, "a") do io
        println(io, "$Rhigh\t$err")
    end

    println("-> Progress saved to disk!")
end

println("\nAll Rhigh values finished!")

writedlm(output_file, [Rhigh_values errs], '\t')
println("Results saved to $output_file")