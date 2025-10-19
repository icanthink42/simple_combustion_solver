import unittest
from flow_state import FlowState

# Based on calculation in https://www.me.psu.edu/cimbala/me320/Lesson_Notes/Fluid_Mechanics_Lesson_15G.pdf
class TestFlowState(unittest.TestCase):
    def setUp(self):
        # Initial conditions from example
        self.state1 = FlowState.from_defined(
            velocity=80,      # m/s
            temp=550,         # K
            pressure=480e3,   # Pa
            density=3.0409,   # kg/m³
            gamma=1.4,        # ratio of specific heats
            c_p=1005,        # J/kg·K
            area=0.017671,    # m² (15-cm diameter tube)
            r_gas=287.05      # J/kg·K
        )

        # Apply heat addition
        self.state2 = self.state1.temperature_rise_from_heat(4514e3)  # 4514 kW

    def test_heat_addition_results(self):
        """Test the results of heat addition match expected values"""
        # Test Mach number
        self.assertAlmostEqual(self.state2.mach, 0.3141876, places=6,
            msg="Mach number after heat addition incorrect")

        # Test temperature
        self.assertAlmostEqual(self.state2.temperature(), 1567.069, places=3,
            msg="Temperature after heat addition incorrect")

        # Test pressure
        self.assertAlmostEqual(self.state2.pressure(), 438896.262, places=3,
            msg="Pressure after heat addition incorrect")

        # Test density
        self.assertAlmostEqual(self.state2.density(), 0.975700, places=6,
            msg="Density after heat addition incorrect")

        # Test velocity
        self.assertAlmostEqual(self.state2.velocity(), 249.331, places=3,
            msg="Velocity after heat addition incorrect")

    def test_mass_conservation(self):
        """Test that mass flow rate is conserved"""
        m_dot1 = self.state1.m_dot()
        m_dot2 = self.state2.m_dot()
        self.assertAlmostEqual(m_dot1, m_dot2, places=6,
            msg="Mass flow rate not conserved")

if __name__ == '__main__':
    unittest.main()
