using HDF5
using Printf

# --- Constants for Primitives ---
const VALID_PRIMS = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]
const USE_GEODESIC_SIGMACUT = true


metric_type = "METRIC_FMKS"
if (metric_type == "METRIC_FMKS")
    poly_xt = 0.82#TODO: prob have to be read from file
    poly_alpha = 14.0#TODO: prob have to be read from file
    mks_smooth = 0.5#TODO: prob have to be read from file
    poly_norm = 0.5 * π * 1. /(1. + 1. /(poly_alpha + 1. )*1. /(poly_xt^poly_alpha));
end

# --- Data Storage Structure ---
"""
    IharmData

A struct to hold the data from an iharm3d simulation dump.
The data is stored in a dictionary where keys are the primitive names
(e.g., "RHO") and values are 3D arrays.
"""
# struct IharmData
#     primitives::Dict{String, Array{Float64, 3}}
# end

struct IharmData
    RHO::Array{Float64,3}
    UU::Array{Float64,3}
    U1::Array{Float64,3}
    U2::Array{Float64,3}
    U3::Array{Float64,3}
    B1::Array{Float64,3}
    B2::Array{Float64,3}
    B3::Array{Float64,3}
    ne::Array{Float64,3}
    b::Array{Float64,3}
    θe::Array{Float64,3}
    sigma::Array{Float64,3}
    beta::Array{Float64,3}
end



# --- Internal Functions ---

function _read_single_primitive(file_handle, prim_name::String)
    if "prims" in keys(file_handle)
        prims = read(file_handle["prims"])
        prim_idx = findfirst(isequal(uppercase(prim_name)), VALID_PRIMS)
        # Returns 'nothing' if the primitive is not found
        return prim_idx === nothing ? nothing : permutedims(prims[prim_idx, :, :, :], (3, 2, 1))
    else
        error("Dataset 'prims' not found in the HDF5 file.")
    end
end


# --- Public Functions ---

"""
    load_data(filename::String) -> IharmData

Loads data from an iharm3d HDF5 dump file.

This function reads the primitive variables (`RHO`, `UU`, etc.) from the specified file,
converts them to `Float64`, and stores them in an `IharmData` object.

# Arguments
- `filename::String`: The path to the HDF5 dump file.

# Returns
- An `IharmData` object containing the simulation data.

# Throws
- An error if the file is not found or if the 'prims' dataset is missing.
"""
# function load_data(filename::String)
#     println("Loading data from '$filename' into 'iharm' module...")
#     !isfile(filename) && error("File not found: $filename")

#     primitives_data = Dict{String, Array{Float64, 3}}()
#     h5open(filename, "r") do file
#         for prim_name in VALID_PRIMS
#             data_3d = _read_single_primitive(file, prim_name)
#             if data_3d !== nothing
#                 # Convert to Float64 to ensure precision
#                 primitives_data[prim_name] = Float64.(data_3d)
#             end
#         end
#     end

#     if isempty(primitives_data)
#         @warn "No primitives were loaded from file '$filename'."
#     else
#         dims = size(primitives_data[RHO])
#         println("Data successfully loaded. Dimensions (N1, N2, N3): $dims")
#     end
#     return IharmData(primitives_data)
# end

function load_data(filename::String, Nfiles::Int = 1)
    println("Loading data from '$filename' into 'iharm' module...")
    !isfile(filename) && error("File not found: $filename")

    # Temporary variables to store primitives
    rho = uu = u1 = u2 = u3 = b1 = b2 = b3 = nothing

    #Nfiles will be useful when using SLOW_LIGHT later on, for now we just load one file
    data_array = Vector{IharmData}(undef, Nfiles)


    #Rescale mdot
    rescale_factor = 1.;
    target_mdot = 0; #TODO: THIS HAS TO BE READ FROM INPUT OR PLACED SOMEWHERE ELSE
    if(target_mdot > 0)
        println("Resetting M_unit to match target_mdot = $target_mdot")
        current_mdot = Mdot_dump/MdotEdd_dump
        println("... is now $(M_unit * abs(target_mdot / current_mdot))")
        rescale_factor = abs(target_mdot / current_mdot)
        M_unit *= rescale_factor
    end


    h5open(filename, "r") do file
        for n in 1:Nfiles
            for prim_name in VALID_PRIMS
                data_3d = _read_single_primitive(file, prim_name)
                if data_3d !== nothing
                    data_3d = Float64.(data_3d)  # ensure Float64

                    # Assign to the correct struct field
                    if prim_name == "RHO"
                        rho = data_3d
                    elseif prim_name == "UU"
                        uu = data_3d
                    elseif prim_name == "U1"
                        u1 = data_3d
                    elseif prim_name == "U2"
                        u2 = data_3d
                    elseif prim_name == "U3"
                        u3 = data_3d
                    elseif prim_name == "B1"
                        b1 = data_3d
                    elseif prim_name == "B2"
                        b2 = data_3d
                    elseif prim_name == "B3"
                        b3 = data_3d
                    end
                end
            end
            data_array[n] = IharmData(rho, uu, u1, u2, u3, b1, b2, b3, zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)))            
        end
    end

    for n in 1:Nfiles
        for i in 1:(N1)
            for j in 1:(N2)
                X::MVec4 = zeros(MVec4)
                ijktoX(i-1, j-1, 0, X)
                gcov = gcov_func(X, bhspin)
                gcon = gcon_func(gcov)
                g = gdet_zone(i-1, j-1, 0)

                r,th = bl_coord(X)
                for k in 1:(N3)
                    ijktoX(i-1, j-1, k, X)

                    Ufields = (data_array[n].U1, data_array[n].U2, data_array[n].U3)
                    UdotU = 0.0
                    for l in 1:(NDIM -1)       # l = 1,2,3 corresponds to U1,U2,U3
                        for m in 1:(NDIM -1)
                            UdotU += gcov[l+1, m+1] * Ufields[l][i,j,k] * Ufields[m][i,j,k]
                        end
                    end

                    ufac = sqrt(-1. / gcon[1,1] * (1. + abs(UdotU)))
                    ucon::MVec4 = MVec4(undef)
                    ucon[1] = -ufac * gcon[1,1]

                    for μ in 1:(NDIM-1)
                        ucon[μ + 1] = Ufields[μ][i,j,k] - ufac * gcon[1, μ+1]
                    end
                    
                    ucov = flip_index(ucon, gcov)
                    udotB = 0.0
                    Bfields = (data_array[n].B1, data_array[n].B2, data_array[n].B3)
                    for l in 1:(NDIM -1)
                        udotB += ucov[l+1] * Bfields[l][i,j,k]
                    end

                    bcon::MVec4 = MVec4(undef)
                    bcon[1] = udotB
                    for μ in 1:(NDIM-1)
                        bcon[μ+1] = (Bfields[μ][i,j,k] + ucon[μ+1] * udotB) / ucon[1]
                    end
                    bcov = flip_index(bcon, gcov)

                    bsq = 0.0
                    for l in 1:NDIM
                        bsq += bcov[l] * bcon[l]
                    end
                    data_array[n].b[i,j,k] = sqrt(bsq) * B_unit  # Magnetic field strength

                end
            end
        end
        init_physical_quantities(data_array, n, rescale_factor)  # Initialize physical quantities for the first dataset
    end

    println("data_array: $(data_array[1].b[1,1,1])")
    println("B_unit: $B_unit")
    # Check if all fields were loaded
    fields = [rho, uu, u1, u2, u3, b1, b2, b3]
    if any(x -> x === nothing, fields)
        @warn "Some primitives were missing in file '$filename'."
    else
        println("All primitives successfully loaded. Dimensions: ", size(rho))
    end

    return data_array
end




"""
    view_slice(slice_data::AbstractArray)

Displays a 1D or 2D slice of data in a formatted way in the terminal. Useful for
quick inspection of simulation data.
"""
function view_slice(slice_data::AbstractArray)
    if ndims(slice_data) > 2
        println("The view_slice function only supports 1D or 2D arrays.")
        return
    end

    println("\n" * "="^60)
    println("Displaying slice with dimensions: ", size(slice_data))
    println("="^60)
    
    # Base.showarray is the internal function Julia uses to display arrays nicely
    Base.showarray(stdout, slice_data, false)
    println()
    println("="^60)
end


function init_physical_quantities(data, n::Int64, rescale_factor::Float64)
    println("Using mixed tp_over_te with trat_small = $(trat_small), trat_large = $(trat_large), and beta_crit = $(beta_crit)")
    rescale_factor = sqrt(rescale_factor)
    for i in 1:(N1)
        for j in 1:(N2)
            for k in 1:(N3)
                data[n].ne[i, j, k] = data[n].RHO[i, j, k] * RHO_unit/(MP + ME) * Ne_factor
                data[n].b[i,j,k] *= rescale_factor

                bsq = data[n].b[i, j, k]/B_unit
                bsq = bsq * bsq
                sigma_m = bsq / (data[n].RHO[i, j, k])
                beta_m = data[n].UU[i, j, k] * (gam - 1.) / (bsq)

                betasq = beta_m * beta_m/ beta_crit/beta_crit
                trat = trat_large * betasq/(1. + betasq) + trat_small/(1. + betasq)
                θe_unit = (MP/ME) * (game - 1.) * (gamp - 1.)/((game - 1.) * trat + (gamp - 1.) )
                data[n].θe[i, j, k] = θe_unit * data[n].UU[i, j, k] / (data[n].RHO[i, j, k])

                data[n].θe[i,j,k] = max(data[n].θe[i,j,k], 1.e-3)
                data[n].sigma[i,j,k] = max(sigma_m, SMALL)
                data[n].beta[i,j,k] = max(beta_m, SMALL)
            end
        end
    end
end

function get_model_sigma(X, data)
    if (X_in_domain(X) == 0)
        return 0.0
    end
    tfac = 0.0 #TODO: when using slowlight, we should implement this
    nA = 1
    nB = 1

    #it should be data[nA].sigma and data[nb].sigma, but since we don't have slowlight yet, we just use nA and nB as 0
    return interp_scalar_time(X, data[nA].sigma, data[nB].sigma, tfac)
end


function get_sigma_smoothfac(sigma)
    sigma_above = sigma_cut
    if(sigma_cut_high > 0.0)
        sigma_above = sigma_cut_high
    end
    if(sigma < sigma_cut)
        return 1.0
    end
    if(sigma > sigma_above)
        return 0.0
    end
    dsig = sigma_above - sigma_cut
    return cos(π/2. /dsig * (sigma - sigma_cut))
end



function get_model_ne(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    sigma_smoothfac = 1.0;

    if(USE_GEODESIC_SIGMACUT)
        sigma = get_model_sigma(X, data);
        if(sigma > sigma_cut)
            return 0.0
        end
        sigma_smoothfac = get_sigma_smoothfac(sigma)
    end

    nA = 1
    nB = 1
    tfac = 0.0 #TODO: when using slowlight, we should implement this
    
    #it should be data[nA].sigma and data[nb].sigma, but since we don't have slowlight yet, we just use nA and nB as 0
    return interp_scalar_time(X, data[nA].ne, data[nB].ne, tfac) * sigma_smoothfac
end

function set_tinterp_ns(X::MVec4)
    """
    How far have we interpolated in time between two data points data[nA] and data[nB]

    Parameters:
    @X: The point in spacetime where we want to interpolate.

    Observations:
    - In slowlight mode, we perform linear interpolation in time. This function tells
    us how far we've progressed from data[nA]->t to data[nB]->t but "in reverse" as
    tinterp == 1 -> we're exactly on nA and tinterp == 0 -> we're exactly on nB. 
    
    - Currently, this function is just a placeholder, it must be implemented if SLOW_LIGHT is true.
    """

    return 0.0, 0, 0
end

function get_model_thetae(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    nA = 1
    nB = 1
    tfac = 0.0 #TODO: when using slowlight, we should implement this

    return interp_scalar_time(X, data[nA].θe, data[nB].θe, tfac)
end

function get_model_b(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    nA = 1
    nB = 1
    tfac = 0.0 #TODO: when using slowlight, we should implement this

    return interp_scalar_time(X, data[nA].b, data[nB].b, tfac)
end

function get_model_fourv(data, X, Kcon, Ucon, Ucov, Bcon, Bcov, bhspin)
    gcov = gcov_func(X, bhspin)
    gcon = gcon_func(gcov)

    if(X_in_domain(X) == 0)
        Ucov[1] = -1. /sqrt(-gcon[1, 1])
        Ucov[2] = 0.0
        Ucov[3] = 0.0
        Ucov[4] = 0.0
        Ucon[1] = 0.0
        Ucon[2] = 0.0
        Ucon[3] = 0.0
        Ucon[4] = 0.0

        for μ in 1:NDIM
            Ucon[1] += Ucov[μ] * gcon[1, μ]
            Ucon[2] += Ucov[μ] * gcon[2, μ]
            Ucon[3] += Ucov[μ] * gcon[3, μ]
            Ucon[4] += Ucov[μ] * gcon[4, μ]
            Bcon[μ] = 0.0
            Bcov[μ] = 0.0
        end
        return
    end
    Vcon = MVec4(undef)
    tfac, _, _ = set_tinterp_ns(X)
    nA = 1 #TODO: when using slowlight, we should implement this
    nB = 1 #TODO: when using slowlight, we should implement this
    Vcon[2] = interp_scalar_time(X, data[nA].U1, data[nB].U1, tfac);
    Vcon[3] = interp_scalar_time(X, data[nA].U2, data[nB].U2, tfac);
    Vcon[4] = interp_scalar_time(X, data[nA].U3, data[nB].U3, tfac);
    VdotV = 0.0
    for μ in 2:NDIM
        for ν in 2:NDIM
            VdotV += gcov[μ, ν] * Vcon[μ] * Vcon[ν]
        end
    end

    Vfac = sqrt(-1. /gcon[1, 1] * (1. + abs(VdotV)))
    Ucon[1] = -Vfac * gcon[1, 1]
    for μ in 2:NDIM
        Ucon[μ] =  Vcon[μ] - Vfac * gcon[1, μ]
    end

    Ucov_local = flip_index(Ucon, gcov)

    #Now set Bcon and get Bcov by lowering it

    Bcon1 = interp_scalar_time(X, data[nA].B1, data[nB].B1, tfac);
    Bcon2 = interp_scalar_time(X, data[nA].B2, data[nB].B2, tfac);
    Bcon3 = interp_scalar_time(X, data[nA].B3, data[nB].B3, tfac);

    Bcon[1] = (Ucon[2] * Bcon1 + Ucon[3] * Bcon2 + Ucon[4] * Bcon3)
    Bcon[2] = (Bcon1 + Ucon[2] * Bcon[1]) / Ucon[1]
    Bcon[3] = (Bcon2 + Ucon[3] * Bcon[1]) / Ucon[1]
    Bcon[4] = (Bcon3 + Ucon[4] * Bcon[1]) / Ucon[1]

    Bcov_local = flip_index(Bcon, gcov)

    for μ in 1:NDIM
        Ucov[μ] = Ucov_local[μ]
        Bcov[μ] = Bcov_local[μ]
    end
end



function radiating_region(X::MVec4, Rh::Float64)
    """
    Checks if the position is within the radiating region.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, _ = bl_coord(X)
    return (r > (Rh + 0.0001) && r > 1. && r < 1000.0)
end