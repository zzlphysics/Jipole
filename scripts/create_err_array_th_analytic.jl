include("../src/main.jl")

using ProgressMeter


const nx, ny = 128,128
dI_dθo_arr = Matrix{Float64}(undef, nx, ny)
dI_da_arr = Matrix{Float64}(undef, nx, ny)
I_arr = Matrix{Float64}(undef, nx, ny)

const DX = DY = 30
const freq = 230e9
const ro = 1000.0
const fovx = fovy = DX / ro
const nmaxstep = 15000
const bhspin = 0.9
const Rout = 1000.0
const Rstop = 10000.0
const th = 163.0
const phi = 0.0

const cstartx = MVec4(0.0, 0.0, 0.0, 0.0)
const cstopx = MVec4(0.0, log(Rout), 1.0, 2.0 * π)
const params = GlobalParams(bhspin, Rout, cstartx, cstopx);

# Pre-allocate trajectory arrays for each thread

using ProgressMeter
# Find camera in native coordinates

Xcamera = MVec4(camera_position(ro, th, phi, params.a, params.Rout))

# Scales the intensity of each pixel by the real size of each pixel
scale_factor = CalculateScaleFactor(DX, DY, nx, ny, 7.778e3 * PC, L_unit)
println("scale_factor = $scale_factor")
const maxnstep = 15000
# Generate geodesics
println("Utilizing $(Threads.nthreads()) threads for geodesic calculation.")

p = Progress(
    nx * ny; 
    desc = "Raytracing Image...", 
    showspeed = true, 
    barlen = 30
)
const Rh = 1 + sqrt(1. - params.a * params.a);
freq_unitless = freq * HPL/(ME * CL * CL) 
Image_truth = zeros(Float64, nx, ny)
@time begin
   Threads.@threads for i in 0:(nx - 1)
        tid = Threads.threadid()
        for j in 0:(ny - 1)
            traj = Vector{OfTraj}()
            sizehint!(traj, maxnstep)
            nstep = get_pixel(traj, i, j, Xcamera, maxnstep, fovx, fovy, freq_unitless, nx, ny, params.a, Rh, params.Rout, Rstop) 
            
            resize!(traj,nstep)
            integrate_emission!(traj, nstep, Image_truth, i + 1, j + 1, freq, params.a)
            ProgressMeter.next!(
                p; 
                showvalues = [
                    (:thread_id, tid), 
                    (:pixel, "($i, $j)"), 
                    (:total_done, "$(i*ny + j)/$(nx * ny)")
                ]
            )
        end
    end
    Image_truth *= freq^3;
end
finish!(p);
println("Image generation complete. Starting error analysis...")
OutputStokesParameters(Image_truth, freq, scale_factor, nx,  7.778e3 * PC)

function measure_error(angle_trying)
    # Find camera in native coordinates

    Xcamera = MVec4(camera_position(ro, angle_trying, phi, params.a, params.Rout))

    # Scales the intensity of each pixel by the real size of each pixel
    println("scale_factor = $scale_factor")
    maxnstep = 15000
    # Generate geodesics
    println("Utilizing $(Threads.nthreads()) threads for geodesic calculation.")

    p = Progress(
        nx * ny; 
        desc = "Raytracing Image...", 
        showspeed = true, 
        barlen = 30
    )
    ProgressMeter.ijulia_behavior(:clear)

    freq_unitless = freq * HPL/(ME * CL * CL) 
    Image = zeros(Float64, nx, ny)
    @time begin
       Threads.@threads for i in 0:(nx - 1)
            tid = Threads.threadid()
            for j in 0:(ny - 1)
                traj = Vector{OfTraj}()
                sizehint!(traj, maxnstep)
                nstep = get_pixel(traj, i, j, Xcamera, maxnstep, fovx, fovy, freq_unitless, nx, ny, params.a, Rh, params.Rout, Rstop) 

                resize!(traj,nstep)
                integrate_emission!(traj, nstep, Image, i + 1, j + 1, freq, params.a)
                ProgressMeter.next!(
                    p; 
                    showvalues = [
                        (:thread_id, tid), 
                        (:pixel, "($i, $j)"), 
                        (:total_done, "$(i*ny + j)/$(nx * ny)")
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

th_trying = collect(1:0.1:189)

# 1. Pre-allocate an array of zeros with the exact same length as th_trying
errs = zeros(Float64, length(th_trying))

output_file = "angle_errors_output_analytic.txt"
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
writedlm("angle_errors_output_analytic.txt", [th_trying errs], '\t')
println("Results saved to angle_errors_output.txt")
