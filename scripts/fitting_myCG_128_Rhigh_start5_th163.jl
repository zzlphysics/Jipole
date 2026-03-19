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
const sigma_cut = 1.e99
println("sigma_cut set to $sigma_cut to effectively ignore it in calculations.")
const sigma_cut_high = -1.0;
const simulation_data = load_data(dump_filepath, trat_large);

#Setting up the parameters
#Observer distance in Rg
const ro = 1000.0
#Observer inclination in degrees
const th = 163.0

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

include("../src/main.jl")
initial_th_guess = 163.0
Rhigh_test = 5.0
ths, Rhighs, err, steps = true_conjugate_gradient_optimization_GRMHD(Image_truth, ro, initial_th_guess, Rhigh_test, freq, nx, ny, nmaxstep, fovx, fovy, params.Rout, Rstop; cost_tol = 2e-3, optimize_param = :Rhigh, dump_filepath = dump_filepath, sensemode = "AD")

println("ths: ", ths)
println("Rhighs: ", Rhighs)
println("err: ", err)
using DelimitedFiles

writedlm("./txts/Fitting/th_mks_128-[ths_Rhighs_err]_AD_lambdagrmonty_sigmacut1e99_from_175_20.txt", [ths Rhighs err], '\t')