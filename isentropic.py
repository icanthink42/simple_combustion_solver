# Ratios - p/pt, T/Tt, rho/rhot
# These give the ratio of static to total (stagnation) properties

def p_p_t(mach, gamma):
    """Pressure ratio p/pt"""
    return 1.0 / ((1 + 0.5 * (gamma - 1) * mach ** 2) ** (gamma / (gamma - 1)))

def t_t_t(mach, gamma):
    """Temperature ratio T/Tt"""
    return 1.0 / (1 + 0.5 * (gamma - 1) * mach ** 2)

def rho_rho_t(mach, gamma):
    """Density ratio rho/rhot"""
    return 1.0 / ((1 + 0.5 * (gamma - 1) * mach ** 2) ** (1.0 / (gamma - 1)))

# Full

def stag_temp(mach, temp, gamma):
    return temp / t_t_t(mach, gamma)

def temp_from_stag(mach, temp, gamma):
    return temp * t_t_t(mach, gamma)

def stag_pressure(mach, pressure, gamma):
    return pressure / p_p_t(mach, gamma)

def pressure_from_stag(mach, pressure, gamma):
    return pressure * p_p_t(mach, gamma)

def stag_density(mach, density, gamma):
    return density / rho_rho_t(mach, gamma)

def density_from_stag(mach, density, gamma):
    return density * rho_rho_t(mach, gamma)

def mdot_from_pressure_drop(init_stag_pressure, exit_pressure, gamma, r_gas, stag_temp, area, c_d=0.95):
    """Calculate mass flow rate through an orifice using compressible flow equation.

    Args:
        init_stag_pressure: Upstream stagnation pressure P₀ (Pa)
        exit_pressure: Downstream pressure Pₑ (Pa)
        gamma: Specific heat ratio γ
        r_gas: Gas constant R (J/kg·K)
        stag_temp: Upstream stagnation temperature T₀ (K)
        area: Flow area A (m²)
        c_d: Discharge coefficient Cᵈ (default 0.95)
    """
    # Calculate pressure ratio Pₑ/P₀
    pressure_ratio = exit_pressure / init_stag_pressure
    critical_pressure_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    if pressure_ratio > critical_pressure_ratio:
        raise RuntimeError(f"Pressure ratio {pressure_ratio} is greater than the critical pressure ratio: {critical_pressure_ratio}")

    # Implement equation:
    # ṁ = Cᵈ A P₀ √(γ/RT₀) (Pₑ/P₀)^(1/γ) √(2/(γ-1)[1-(Pₑ/P₀)^((γ-1)/γ)])
    mdot = (c_d * area * init_stag_pressure *
            (gamma / (r_gas * stag_temp))**0.5 *
            pressure_ratio**(1/gamma) *
            (2/(gamma-1) * (1 - pressure_ratio**((gamma-1)/gamma)))**0.5)

    return mdot