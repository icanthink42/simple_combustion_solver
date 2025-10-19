import unittest
import isentropic

class TestIsentropic(unittest.TestCase):
    def setUp(self):
        # Common test values
        self.gamma = 1.4  # Air
        self.mach_subsonic = 0.5
        self.mach_sonic = 1.0
        self.mach_supersonic = 2.0

        # Known ratios for M = 0.5 with gamma = 1.4 from compressible flow tables
        self.known_p_pt_M05 = 0.8430  # p/pt at M=0.5
        self.known_t_tt_M05 = 0.9524  # T/Tt at M=0.5
        self.known_rho_rhot_M05 = 0.8852  # rho/rhot at M=0.5 (rounded to 4 decimal places)

        # Known ratios for M = 2.0 with gamma = 1.4
        self.known_p_pt_M20 = 0.1278  # p/pt at M=2.0
        self.known_t_tt_M20 = 0.5556  # T/Tt at M=2.0
        self.known_rho_rhot_M20 = 0.2300  # rho/rhot at M=2.0

        # Values from Rayleigh flow example
        self.example_mach = 0.17018  # M1 from example
        self.example_temp = 550  # K
        self.example_pressure = 480e3  # Pa
        self.example_density = 3.0409  # kg/m³

    def test_pressure_ratio_subsonic(self):
        """Test pressure ratio at M=0.5"""
        ratio = isentropic.p_p_t(self.mach_subsonic, self.gamma)
        self.assertAlmostEqual(ratio, self.known_p_pt_M05, places=4)

    def test_pressure_ratio_supersonic(self):
        """Test pressure ratio at M=2.0"""
        ratio = isentropic.p_p_t(self.mach_supersonic, self.gamma)
        self.assertAlmostEqual(ratio, self.known_p_pt_M20, places=4)

    def test_temperature_ratio_subsonic(self):
        """Test temperature ratio at M=0.5"""
        ratio = isentropic.t_t_t(self.mach_subsonic, self.gamma)
        self.assertAlmostEqual(ratio, self.known_t_tt_M05, places=4)

    def test_temperature_ratio_supersonic(self):
        """Test temperature ratio at M=2.0"""
        ratio = isentropic.t_t_t(self.mach_supersonic, self.gamma)
        self.assertAlmostEqual(ratio, self.known_t_tt_M20, places=4)

    def test_density_ratio_subsonic(self):
        """Test density ratio at M=0.5"""
        ratio = isentropic.rho_rho_t(self.mach_subsonic, self.gamma)
        self.assertAlmostEqual(ratio, self.known_rho_rhot_M05, places=4)

    def test_density_ratio_supersonic(self):
        """Test density ratio at M=2.0"""
        ratio = isentropic.rho_rho_t(self.mach_supersonic, self.gamma)
        self.assertAlmostEqual(ratio, self.known_rho_rhot_M20, places=4)

    def test_sonic_conditions(self):
        """Test that ratios at M=1 give expected values"""
        self.assertAlmostEqual(isentropic.p_p_t(self.mach_sonic, self.gamma), 0.5283, places=4)
        self.assertAlmostEqual(isentropic.t_t_t(self.mach_sonic, self.gamma), 0.8333, places=4)
        self.assertAlmostEqual(isentropic.rho_rho_t(self.mach_sonic, self.gamma), 0.6339, places=4)

    def test_stagnation_recovery(self):
        """Test that converting to stagnation and back gives original value"""
        # Test with temperature
        temp = 300  # K
        stag_temp = isentropic.stag_temp(self.mach_subsonic, temp, self.gamma)
        recovered_temp = isentropic.temp_from_stag(self.mach_subsonic, stag_temp, self.gamma)
        self.assertAlmostEqual(temp, recovered_temp, places=10)

        # Test with pressure
        pressure = 101325  # Pa
        stag_pressure = isentropic.stag_pressure(self.mach_subsonic, pressure, self.gamma)
        recovered_pressure = isentropic.pressure_from_stag(self.mach_subsonic, stag_pressure, self.gamma)
        self.assertAlmostEqual(pressure, recovered_pressure, places=10)

        # Test with density
        density = 1.225  # kg/m³
        stag_density = isentropic.stag_density(self.mach_subsonic, density, self.gamma)
        recovered_density = isentropic.density_from_stag(self.mach_subsonic, stag_density, self.gamma)
        self.assertAlmostEqual(density, recovered_density, places=10)

    def test_rayleigh_example_values(self):
        """Test with values from the Rayleigh flow example"""
        # Calculate stagnation conditions
        stag_t = isentropic.stag_temp(self.example_mach, self.example_temp, self.gamma)
        stag_p = isentropic.stag_pressure(self.example_mach, self.example_pressure, self.gamma)
        stag_rho = isentropic.stag_density(self.example_mach, self.example_density, self.gamma)

        # Convert back to static conditions
        temp = isentropic.temp_from_stag(self.example_mach, stag_t, self.gamma)
        pressure = isentropic.pressure_from_stag(self.example_mach, stag_p, self.gamma)
        density = isentropic.density_from_stag(self.example_mach, stag_rho, self.gamma)

        # Verify recovery of original values
        self.assertAlmostEqual(temp, self.example_temp, places=10)
        self.assertAlmostEqual(pressure, self.example_pressure, places=10)
        self.assertAlmostEqual(density, self.example_density, places=10)

if __name__ == '__main__':
    unittest.main()
