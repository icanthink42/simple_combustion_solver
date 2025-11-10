import isentropic
from scipy.optimize import root_scalar
import numpy as np
# https://www.me.psu.edu/cimbala/me320/Lesson_Notes/Fluid_Mechanics_Lesson_15G.pdf

class FlowState:
    def __init__(self, mach, stag_temp, stag_pressure, stag_density, gamma, c_p, area, r_gas):
        self.mach = mach
        self.stagnation_temperature = stag_temp
        self.stagnation_pressure = stag_pressure
        self.stagnation_density = stag_density
        self.gamma = gamma
        self.c_p = c_p
        self.area = area
        self.r_gas = r_gas

    @staticmethod
    def from_defined(velocity, temp, pressure, density, gamma, c_p, area, r_gas):

        mach = velocity / (gamma * r_gas * temp)**0.5

        stag_temp = isentropic.stag_temp(mach, temp, gamma)
        stag_pressure = isentropic.stag_pressure(mach, pressure, gamma)
        stag_density = isentropic.stag_density(mach, density, gamma)

        return FlowState(mach, stag_temp, stag_pressure, stag_density, gamma, c_p, area, r_gas)

    @staticmethod
    def from_defined_mdot(mdot, temp, pressure, density, gamma, c_p, area, r_gas):
        velocity = mdot / (density * area)
        return FlowState.from_defined(velocity, temp, pressure, density, gamma, c_p, area, r_gas)

    def temperature(self):
        return isentropic.temp_from_stag(self.mach, self.stagnation_temperature, self.gamma)

    def pressure(self):
        return isentropic.pressure_from_stag(self.mach, self.stagnation_pressure, self.gamma)

    def density(self):
        return isentropic.density_from_stag(self.mach, self.stagnation_density, self.gamma)

    def velocity(self):
        return self.mach * (self.gamma * self.r_gas * self.temperature()) ** 0.5

    def stagnation_enthalpy(self):
        return self.c_p * self.stagnation_temperature + self.velocity()**2 / 2

    def enthalpy(self):
        stag_enthalpy = self.c_p * self.stagnation_temperature + self.velocity()**2 / 2
        return isentropic.enthalpy_from_stag(self.mach, stag_enthalpy, self.gamma)

    def m_dot(self):
        return self.density() * self.velocity() * self.area

    def _find_mach2(self, t02_t0_star):

        def equation(m2):
            num = (2 + (self.gamma - 1) * m2**2) * ((1 + self.gamma) * m2**2)
            den = (1 + self.gamma * m2**2)**2
            return num/den - t02_t0_star

        if t02_t0_star > 1:
            raise RuntimeError(f"t02_t0_star = {t02_t0_star} is greater than 1. Supersonic flow is not supported. Failure velocity is {self.velocity()} m/s.")

        result = root_scalar(equation, x0=self.mach, method='newton')

        return result.root

    def temperature_rise_from_heat(self, q_dot):
        # Calculate mass flow rate and specific heat addition
        m_dot = self.m_dot()
        q = q_dot / m_dot

        # New stagnation temperature
        new_stag_temp = self.stagnation_temperature + q / self.c_p

        # Calculate T01/T0* (from example Step 2)
        t01_t0_star = ((2 + (self.gamma - 1) * self.mach**2) * ((1 + self.gamma) * self.mach**2) /
                      (1 + self.gamma * self.mach**2)**2)

        # Calculate T02/T0* (from example Step 3)
        t02_t0_star = (new_stag_temp / self.stagnation_temperature) * t01_t0_star

        new_mach = self._find_mach2(t02_t0_star)

        # Calculate new static temperature using isentropic relations
        new_static_temp = isentropic.temp_from_stag(new_mach, new_stag_temp, self.gamma)

        # Calculate new static density to maintain mass flow rate
        new_static_velocity = new_mach * (self.gamma * self.r_gas * new_static_temp)**0.5
        new_static_density = m_dot / (new_static_velocity * self.area)

        # Calculate new stagnation conditions using isentropic relations
        new_stag_density = isentropic.stag_density(new_mach, new_static_density, self.gamma)
        new_stag_pressure = new_stag_density * self.r_gas * new_stag_temp

        return FlowState(new_mach, new_stag_temp, new_stag_pressure, new_stag_density, self.gamma, self.c_p, self.area, self.r_gas)

    def temperature_rise_from_combustion(self, mdot_fuel, LHV):
        q_dot = LHV * mdot_fuel
        return self.temperature_rise_from_heat(q_dot)

    def compressor(self, pressure_ratio, thermal_efficiency):
        new_stag_pressure = self.stagnation_pressure * pressure_ratio
        new_stag_temp = self.stagnation_temperature * (pressure_ratio)**((self.gamma - 1)/(self.gamma * thermal_efficiency))
        new_stag_density = new_stag_pressure / (self.r_gas * new_stag_temp)

        # Preserve mass flow rate and area; solve for new Mach from compressible flow MFP
        target_mdot = self.m_dot()
        gamma = self.gamma
        r_gas = self.r_gas
        area = self.area

        def mass_flow_parameter(m):
            return m * (1 + 0.5 * (gamma - 1) * m**2) ** (-(gamma + 1) / (2 * (gamma - 1)))

        constant = area * new_stag_pressure * (gamma / (r_gas * new_stag_temp)) ** 0.5

        def f(m):
            return constant * mass_flow_parameter(m) - target_mdot

        # Check for choking (no subsonic solution if required mdot exceeds choked mdot)
        mdot_choked = constant * mass_flow_parameter(1.0)
        if target_mdot > mdot_choked:
            raise RuntimeError("Required mass flow exceeds choked mass flow for the given compressor exit conditions")

        # Find a subsonic root in (0, 1)
        result = root_scalar(f, x0=0.5, method='newton')
        new_mach = result.root
        print("new_mach", new_mach)
        print(result.converged)

        return FlowState(new_mach, new_stag_temp, new_stag_pressure, new_stag_density, self.gamma, self.c_p, self.area, self.r_gas)

    def compressor_pressure_increase(self, enthalpy_increase):

        new_stag_enthalpy = self.enthalpy() + enthalpy_increase


def main():
    state1 = FlowState.from_defined(velocity=80, temp=550, pressure=480e3, density=3.0409, gamma=1.4, c_p=1005, area=0.017671, r_gas=287.05)
    state2 = state1.temperature_rise_from_heat(4514e3)
    print("mach", state2.mach)
    print("temperature", state2.temperature())
    print("pressure", state2.pressure())
    print("density", state2.density())
    print("velocity", state2.velocity())

if __name__ == '__main__':
    main()