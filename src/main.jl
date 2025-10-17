using Printf
using Base.Threads
using LinearAlgebra
using StaticArrays
using ForwardDiff

#Mutable 4-dimensional vector allocated on the stack
const MVec4  = MVector{4,Float64}

#
const TMVec4{T} = MVector{4,T}

#Mutable 4x4 matrix allocated on the stack
const MMat4 = MMatrix{4,4,Float64}

const TMMat4{T} = MMatrix{4,4,T,16}

#Mutable 3-dimensional vector allocated on the stack
const Tensor3D = MArray{Tuple{4,4,4}, Float64, 3, 64}  # 4×4×4 mutable tensor
const TTensor3D{T} = MArray{Tuple{4,4,4}, T, 3, 64}  # 4×4×4 mutable tensor



mutable struct OfTraj
    dl::Float64
    X::MVec4
    Kcon::MVec4
    Xhalf::MVec4
    Kconhalf::MVec4
    dX_dθo::MVec4
    dK_dθo::MVec4
    dX_da::MVec4
    dK_da::MVec4
end
include("constants.jl")
include("set_globals.jl")
include("camera.jl")
include("debug_functions.jl")
include("metrics.jl")
include("coords.jl")
include("tetrads.jl")
include("utils.jl")
include("radiation.jl")
#include("./models/$(MODEL).jl")

##MODELS
include("./models/$(MODEL).jl")
#1) iharm
using .iharm
#include("/home/raia/Jipole/src/models/iharm.jl")


include("geodesics.jl")
include("autodiff.jl")
include("gradientdescent.jl")


# function calcKcon(sr::Bool, sθ::Bool, r::Float64, θ::Float64, ϕ::Float64, metric::Kerr, η::Float64, λ::Float64, nstep::Int)
#     """
#     Calculates the covariant 4-velocity of the photon for each point in the geodesics in internal coordinates.
    
#     Observations:
#     - We follow the equation 9 of Gelles et al. 2021 https://arxiv.org/pdf/2105.09440
#     - The function Krang.p_bl_d returns the wavevector in BL coordinates, so we transform for the native coordinates.
#     """
#     X = MVec4(0, log(r), θ/π, ϕ)
#     Kcovbl = Krang.p_bl_d(metric, r, θ, η, λ, sr, sθ) 
#     Kcovbl *= HPL * 230e9/(ME * CL * CL)

#     #Transform Kcov to Kcon
#     bl_gcov = gcov_bl(r, θ)
#     bl_gcon = gcon_func(bl_gcov)
#     KconBL = flip_index(MVec4(Kcovbl), bl_gcon)

#     #Transform from BL to KS coordinates
#     KconKS = bl_to_ks(X, KconBL)

#     #Now convert to Native Coordinates
#     KconNC = vec_from_ks(X, KconKS)
#     return KconNC
# end



# function calculate_intensity_krang(scale_factor_ipole::Float64, rad_fovx::Float64, rad_fovy::Float64, camera_position::MVec4, pixels_x, pixels_y, freq_cgs)
#     Image =  zeros(Float64, pixels_x, pixels_y)
#     radialcam = camera_position[2]
#     θcam = camera_position[3]
#     metric = Krang.Kerr(a);
#     θo = θcam * π / 180;
#     dα = (αmax - αmin) / (pixels_x)
#     dβ = (βmax - βmin) / (pixels_y)
#     #scale_factor = dα * dβ * L_unit^2 / (Dsource * Dsource) / JY
#     scale_factor = scale_factor_ipole
#     res = pixels_x
#     println("Using krang for geodesics calculations, this may take a while...")

#     camera = Krang.IntensityCamera(metric, θo, radialcam, -rad_fovx, rad_fovx, -rad_fovy, rad_fovy, res);
#     #camera = Krang.IntensityCamera(metric, θo, αmin, αmax, βmin, βmax, res);
#     lines = Krang.generate_ray.(camera.screen.pixels, krang_points)


#     for idx in eachindex(lines)
#         Intensity= 0.0
#         I = div(idx - 1, res) + 1
#         J = mod(idx - 1, res) + 1
#         if J == 1
#             println("Processing row $I out of $(res)")
#         end
#         line = lines[idx]
#         nstep = length(line)

#         t = [pt.ts for pt in line]
#         r = [pt.rs for pt in line]
#         th = [pt.θs for pt in line]
#         phi = [pt.ϕs for pt in line]
#         #These are BL, I have to convert to KS coordinates
#         KrangconKS = [bl_to_ks(MVec4(0, log(r[k]), th[k]/π, 0), MVec4(t[k], r[k], th[k], phi[k])) for k in 1:nstep if r[k] >= Rh]
#         #record which index is maximum valid and save as last_valid_index
#         last_valid_index = findlast(k -> k >= Rh, r)

#         KrangNC = [vec_from_ks(MVec4(0, log(r[k]), th[k]/π, 0), KrangconKS[k]) for k in 1:last_valid_index]


#         #check if any r is inf or nan, in case it is, print the r array index where it happens
#         if any(isnan.(r)) || any(isinf.(r))
#             @error "NaN or Inf encountered in r at pixel ($I, $J)"
#             for k in eachindex(r)
#                 if isnan(r[k]) || isinf(r[k])
#                     @error "NaN or Inf encountered in r at index $k: r[$k] = $(r[k])"
#                 end
#             end
#             error("NaN or Inf encountered in r")
#         end

#         sr = [pt.νr for pt in line]      
#         sθ = [pt.νθ for pt in line]
#         Kcon = zeros(Float64, nstep, NDIM)
#         dl = zeros(Float64, nstep)

#         η = Krang.η(camera.screen.pixels[idx])
#         λ = Krang.λ(camera.screen.pixels[idx])

#         for k in 1:last_valid_index
#             if(KrangconKS[k][2] == 0)
#                 break
#             end
#             if (KrangconKS[k][2] < (Rh + 0.5 ))
#                 r[k] = 0.0
#                 break
#             end
#             Kcon[k, :] = calcKcon(sr[k], sθ[k], r[k], th[k], phi[k], metric, η, λ, k)
#         end

#         for k in 1:last_valid_index-1
#             dl[k] = sqrt(sum((KrangNC[k+1][i] - KrangNC[k][i])^2 for i in 1:4))
#         end

        

#         traj = [OfTraj(dl[k], MVec4(KrangNC[k][1], KrangNC[k][2], KrangNC[k][3], KrangNC[k][4]),
#                             MVec4(Kcon[k,1], Kcon[k,2], Kcon[k,3], Kcon[k,4]),
#                             MVec4(undef), MVec4(undef)) for k in 1:last_valid_index]
#         integrate_emission!(traj, last_valid_index, Image, I, J)
#     end


#     Ftot::Float64 = 0.0
#     Iavg::Float64 = 0.0
#     Imax::Float64 = 0.0
#     imax::Int = 0
#     jmax::Int = 0
#     println("Image processing complete. Calculating total flux and averages...")
#     for i in 1:pixels_x
#         for j in 1:pixels_y
#             Ftot += Image[i, j] * freq_cgs^3 * scale_factor
#             Iavg += Image[i, j]
#             if (Image[i,j] * freq_cgs^3) > Imax
#                 imax = i
#                 jmax = j
#                 Imax = Image[i, j] * freq_cgs^3
#             end
#         end
#     end
#     Iavg *= freq_cgs^3/ (pixels_x * pixels_y)
#     @printf("Scale = %.15e\n", scale_factor)
#     println("imax = $imax, jmax = $jmax, Imax = $Imax, Iavg = $Iavg")
#     println("Using freq_cgs = $freq_cgs, Ftot = $(Ftot)")
#     println("nuLnu = $(Ftot * Dsource * Dsource * JY * freq_cgs * 4.0 * π)")
#     return Image* freq_cgs^3 * scale_factor
# end



function IpoleGeoIntensityIntegration(traj, freq_cgs::Float64, nx::Int,ny::Int, scalefactor::Float64, bhspin::Float64)
    """
    Once the trajectories are calculated, integrate the intensity for each pixel.
    Parameters:
    @traj: Matrix of geodesic trajectories for each pixel.
    @freq_cgs: Frequency in cgs units.
    @res: Resolution of the image (number of pixels).
    @scalefactor: Scale factor for the image intensity.

    Returns:
    A matrix representing the integrated intensity for each pixel in the image.
    """
    Image = zeros(Float64, nx, ny)
    Threads.@threads for i in 0:(nx - 1)
        println("Processing row $i out of $(nx)")
        for j in 0:(ny - 1)
            integrate_emission!(traj[i+1, j+1], length(traj[i+1, j+1]), Image, i + 1, j + 1, freq_cgs, bhspin)
        end
    end

    return (Image * freq_cgs^3)
end

function OutputStokesParameters(Image, freq_cgs, scale_factor, res, Dsource)
    println("Image processing complete. Calculating total flux and averages...")
    Ftot::Float64 = 0.0
    Iavg::Float64 = 0.0
    Imax::Float64 = 0.0
    imax::Int = 0
    jmax::Int = 0
    for i in 1:res
        for j in 1:res
            Ftot += Image[i, j] * scale_factor
            Iavg += Image[i, j]
            if (Image[i,j]) > Imax
                imax = i
                jmax = j
                Imax = Image[i, j]
            end
        end
    end
    Iavg *= 1.0/ (res * res)
    @printf("Scale = %.15e\n", scale_factor)
    println("imax = $imax, jmax = $jmax, Imax = $Imax, Iavg = $Iavg")
    println("Using freq_cgs = $freq_cgs, Ftot = $Ftot")
    println("nuLnu = $(Ftot * Dsource * Dsource * JY * freq_cgs * 4.0 * π)")
end

function CalculateScaleFactor(sizex, sizey, pixelsx, pixelsy, SourceD, LengthUnit)
    """
    Calculate the scale factor for the image based on the camera parameters and the source distance.

    Parameters:
    @sizex: Size of the image in the x direction in Rg.
    @sizey: Size of the image in the y direction in Rg.
    @pixelsx: Number of pixels in the x direction.
    @pixelsy: Number of pixels in the y direction.
    @SourceD: Distance to the source in cm.
    @LengthUnit: Length unit in cm (e.g., Rg).
    
    Returns:
    A scalar scale factor for the image intensity.
    """
    return (sizex * LengthUnit / pixelsx) * (sizey * LengthUnit / pixelsy) / (SourceD * SourceD) / JY

end


function main()
    @time begin
        check_parameters()
        println("Generating an image with size $(nx) x $(ny) pixels")
        println("Model Parameters: A = $A, α = $α_analytic, height = $height, l0 = $l0, a = $a")
        println("MBH = $MBH, L_unit = $L_unit")
        println("Dsource = $Dsource")

        Xcam = camera_position(rcam, thcam, phicam, Rout)

        fovx = DX/(rcam)
        fovy = DY/(rcam) 

        scale_factor = CalculateScaleFactor(DXsize, DYsize, nx, ny, SourceD, L_unit)
        println("Running on ", Threads.nthreads(), " threads")

        
        trajectory = CalculateGeodesics(Xcam, fovx, fovy, freqcgs, maxnstep, nx, ny)
        Image =  zeros(Float64, nx, ny)
        Image = IpoleGeoIntensityIntegration(trajectory, freqcgs, nx,ny, scale_factor)
    end
end


#main()
#GC.gc()

