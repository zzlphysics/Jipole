
const MODEL = "iharm" # Model type: "analytic", "thin_disk", "iharm"


# Mass of the black hole in solar masses
const MBH = 6.2e9#4.063e6
const L_unit = GNEWT * MBH * MSUN / CL^2  # Length unit in gravitational radius (Rg)
const T_unit = L_unit / CL  # Time unit in seconds
const M_unit = 3e26
const RHO_unit = M_unit / L_unit^3  # Density unit in g/cm^3
const U_unit = RHO_unit * CL^2  # Internal energy density unit in erg
const B_unit = CL * sqrt(4 * π * RHO_unit)  # Magnetic field unit in Gauss
