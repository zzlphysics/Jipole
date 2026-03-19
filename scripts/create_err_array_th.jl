include("../src/main.jl")
println("Available threads: ", nthreads())
println("RHO_unit:", RHO_unit)
dump_filepath = "../sample_dump_SANE_a+0.94_MKS_0900.h5";
const params = read_header(dump_filepath);
#TODO: put this in reading file
trat_large = 20. 
const trat_small = 1. 
const beta_crit = 1.0 
const th_beg = 1.74e-2 
const sigma_cut = 1.
const sigma_cut_high = -1.0;

const simulation_data = load_data(dump_filepath, trat_large);

#Setting up the parameters
#Observer distance in Rg
const ro = 1000.0
#Observer inclination in degrees
const th = 90.0

#Observer azimuth in degrees
const phi = 0.0

# Number of pixels in the x and y direction. The number of geodesics calculated will be res^2
const pixels_x = 128
const pixels_y = 128
# Distance to the source in parsecs
const SourceD = 16.9e6 * PC
const Rstop = 100.0
const Rh = 1 + sqrt(1. - params.a * params.a);

# Frequency observed by the camera in Hz
const freq = 230e9;

# Size of the screen in Rg in both directions
const DXsize = SourceD/L_unit/MUAS_PER_RAD * 160
const DYsize = SourceD/L_unit/MUAS_PER_RAD * 160
# Observer fov in radians (this can be understood as size of the plane camera sees over the distance ro)
# This should be atan, but for small angles it is approximately equal to the angle itself
const fovx = DXsize/ro
const fovy = DYsize/ro
const xoff = 0.0
const yoff = 0.0
const nmaxstep = 15000
const nx, ny = pixels_x, pixels_y

using ProgressMeter
# Find camera in native coordinates

Xcamera = MVec4(camera_position(ro, th, phi, params.a, params.Rout))

# Scales the intensity of each pixel by the real size of each pixel
scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)
println("scale_factor = $scale_factor")
const maxnstep = 15000
# Generate geodesics
println("Utilizing $(Threads.nthreads()) threads for geodesic calculation.")

p = Progress(
    pixels_x * pixels_y; 
    desc = "Raytracing Image...", 
    showspeed = true, 
    barlen = 30
)
freq_unitless = freq * HPL/(ME * CL * CL) 
Image_truth = zeros(Float64, pixels_x, pixels_y)
@time begin
   Threads.@threads for i in 0:(pixels_x - 1)
        tid = Threads.threadid()
        for j in 0:(pixels_y - 1)
            traj = Vector{OfTraj}()
            sizehint!(traj, maxnstep)
            nstep = get_pixel(traj, i, j, Xcamera, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, params.a, Rh, params.Rout, Rstop, xoff, yoff) 
            
            resize!(traj,nstep)
            integrate_emission!(traj, nstep, Image_truth, i + 1, j + 1, freq, params.a, simulation_data)
            ProgressMeter.next!(
                p; 
                showvalues = [
                    (:thread_id, tid), 
                    (:pixel, "($i, $j)"), 
                    (:total_done, "$(i*pixels_y + j)/$(pixels_x * pixels_y)")
                ]
            )
        end
    end
    Image_truth *= freq^3;
end
finish!(p);

function measure_error(angle_trying)
    # Find camera in native coordinates

    Xcamera = MVec4(camera_position(ro, angle_trying, phi, params.a, params.Rout))

    # Scales the intensity of each pixel by the real size of each pixel
    scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)
    println("scale_factor = $scale_factor")
    maxnstep = 15000
    # Generate geodesics
    println("Utilizing $(Threads.nthreads()) threads for geodesic calculation.")

    p = Progress(
        pixels_x * pixels_y; 
        desc = "Raytracing Image...", 
        showspeed = true, 
        barlen = 30
    )
    ProgressMeter.ijulia_behavior(:clear)

    freq_unitless = freq * HPL/(ME * CL * CL) 
    Image = zeros(Float64, pixels_x, pixels_y)
    @time begin
       Threads.@threads for i in 0:(pixels_x - 1)
            tid = Threads.threadid()
            for j in 0:(pixels_y - 1)
                traj = Vector{OfTraj}()
                sizehint!(traj, maxnstep)
                nstep = get_pixel(traj, i, j, Xcamera, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, params.a, Rh, params.Rout, Rstop, xoff, yoff) 

                resize!(traj,nstep)
                integrate_emission!(traj, nstep, Image, i + 1, j + 1, freq, params.a, simulation_data)
                ProgressMeter.next!(
                    p; 
                    showvalues = [
                        (:thread_id, tid), 
                        (:pixel, "($i, $j)"), 
                        (:total_done, "$(i*pixels_y + j)/$(pixels_x * pixels_y)")
                    ]
                )
            end
        end
        Image *= freq^3;
    end
    finish!(p);
    
    err = cost_func(Image_truth, Image)
    return err

end

th_trying = collect(1:1:179)

# 1. Pre-allocate an array of zeros with the exact same length as th_trying
errs = zeros(Float64, length(th_trying))

output_file = "angle_errors_output_MKS_90_128.txt"
open(output_file, "w") do io
    println(io, "Theta\tError")
end
println("Starting evaluation of $(length(th_trying)) angles. Results will stream to $output_file...")
# 2. Loop through each index
for k in 1:length(th_trying)
    angle = th_trying[k]
    
    println("\n========================================")
    println("Evaluating angle $k/$(length(th_trying)): θ = $angle")
    println("========================================")
    
    # Calculate the error
    err = measure_error(angle)
    errs[k] = err
    
    println("-> Error for θ = $angle is: $err")
    
    # 4. APPEND the new data to the file immediately (using "a")
    open(output_file, "a") do io
        println(io, "$angle\t$err")
    end
    
    println("-> Progress saved to disk!")
end

println("\nAll angles finished! Saving results...")

# 3. Save the results to a file immediately so you don't lose them!
using DelimitedFiles
writedlm("angle_errors_output_MKS_90_128.txt", [th_trying errs], '\t')
println("Results saved to angle_errors_output_MKS_90_128.txt")
