from flow_state import FlowState
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys


def calculate_heat_transfer_coefficient(state, hydraulic_diameter):
    """
    Calculate convective heat transfer coefficient using Dittus-Boelter correlation.

    Args:
        state: FlowState object with flow properties
        hydraulic_diameter: Hydraulic diameter (m)

    Returns:
        h: Heat transfer coefficient (W/m²·K)
    """
    # Get flow properties
    temp = state.temperature()  # K
    velocity = state.velocity()  # m/s
    density = state.density()  # kg/m³

    # Estimate gas properties as function of temperature (for combustion products)
    # Dynamic viscosity (Sutherland's law approximation)
    mu_ref = 1.716e-5  # Pa·s at 273K
    T_ref = 273.15  # K
    S = 110.4  # Sutherland constant for air, K
    mu = mu_ref * (temp / T_ref)**1.5 * (T_ref + S) / (temp + S)  # Pa·s

    # Thermal conductivity (approximate for high-temp gases)
    k_gas = 0.0241 * (temp / 273.15)**0.9  # W/m·K (rough approximation)

    # Prandtl number for combustion gases (typically ~0.7-0.75)
    Pr = state.c_p * mu / k_gas

    # Reynolds number
    Re = density * velocity * hydraulic_diameter / mu

    # Dittus-Boelter correlation for turbulent flow (Re > 10000)
    # Nu = 0.023 * Re^0.8 * Pr^0.4 (heating)
    if Re > 10000:
        Nu = 0.023 * Re**0.8 * Pr**0.4
    else:
        # Laminar flow (conservative estimate)
        Nu = 3.66

    # Heat transfer coefficient
    h = Nu * k_gas / hydraulic_diameter

    return h


def main(input_data):
    # Parse input JSON
    params = json.loads(input_data)

    # Parse all parameters
    try:
        # Required parameters
        mdot_min = float(params['mdot_min'])
        mdot_max = float(params['mdot_max'])
        mdot_samples = int(params['mdot_samples'])
        input_temp = float(params['T_in'])  # K
        input_pressure = float(params['P_in'])  # Pa
        afr = float(params['fuel_air_ratio'])
        outer_radius = float(params['outer_radius'])  # m
        inner_radius = float(params['inner_radius'])  # m
        wall_thickness = float(params['wall_thickness'])  # m
        wall_conductivity = float(params['wall_conductivity'])  # W/m·K

        # Optional parameters with defaults
        external_temp = float(params.get('external_temp', 300))  # K (ambient/cooling temperature)
        h_external = float(params.get('h_external', 100))  # W/m²·K (external heat transfer coefficient)
        combustion_efficiency = float(params.get('combustion_efficiency', 1.0))
        gamma = float(params.get('gamma', 1.4))
        c_p = float(params.get('c_p', 1005))  # J/kg·K
        r_gas = float(params.get('r_gas', 287.05))  # J/kg·K
        kerosene_LHV = float(params.get('kerosene_LHV', 43.1e6))  # J/kg
    except (KeyError, ValueError) as e:
        return f"Error: Invalid parameter - {str(e)}"

    # Generate mass flow rate range
    mdot = np.linspace(mdot_min, mdot_max, mdot_samples)

    input_density = input_pressure / (r_gas * input_temp) # kg/m³
    area = np.pi * (outer_radius**2 - inner_radius**2)  # Annular cross-section area

    # Calculate hydraulic diameter for annular duct: D_h = 4*A/P
    # For annular: D_h = 2*(r_outer - r_inner)
    hydraulic_diameter = 2 * (outer_radius - inner_radius)  # m

    # Arrays for stagnation conditions
    stag_temp = np.zeros(len(mdot))
    stag_pressure = np.zeros(len(mdot))
    stag_density = np.zeros(len(mdot))

    # Arrays for static conditions
    static_temp = np.zeros(len(mdot))
    static_pressure = np.zeros(len(mdot))
    static_density = np.zeros(len(mdot))

    exit_velocity = np.zeros(len(mdot))
    mach = np.zeros(len(mdot))
    h_gas_array = np.zeros(len(mdot))  # Store calculated h_gas values

    failure_index = None
    for i in range(len(mdot)):
        state = FlowState.from_defined_mdot(mdot[i], temp=input_temp, pressure=input_pressure, density=input_density, gamma=gamma, c_p=c_p, area=area, r_gas=r_gas)
        try:
            state = state.temperature_rise_from_combustion(state.m_dot() * afr, kerosene_LHV * combustion_efficiency)
        except RuntimeError:
            failure_index = i
            if failure_index == 0:
                raise RuntimeError("Failure at first inlet pressure")
            break
        stag_temp[i] = state.stagnation_temperature
        stag_pressure[i] = state.stagnation_pressure
        stag_density[i] = state.stagnation_density
        exit_velocity[i] = state.velocity()
        mach[i] = state.mach

        # Calculate heat transfer coefficient from flow properties
        h_gas_array[i] = calculate_heat_transfer_coefficient(state, hydraulic_diameter)

    if failure_index is not None:
        mdot = mdot[:failure_index]
        stag_temp = stag_temp[:failure_index]
        stag_pressure = stag_pressure[:failure_index]
        stag_density = stag_density[:failure_index]
        exit_velocity = exit_velocity[:failure_index]
        mach = mach[:failure_index]
        h_gas_array = h_gas_array[:failure_index]

    # Calculate wall temperature distribution for the outer wall
    # Using radial heat conduction through cylindrical wall
    wall_samples = 50
    r_inner_wall = outer_radius  # Inner surface of wall (hot gas side)
    r_outer_wall = outer_radius + wall_thickness  # Outer surface (external side)

    # Create radial positions through the wall
    r_wall = np.linspace(r_inner_wall, r_outer_wall, wall_samples)

    # Convert radial positions to distance through wall thickness for easier visualization
    wall_distance = (r_wall - r_inner_wall) * 1000  # Convert to mm

    # Calculate wall temperature for all mass flow rates (for 3D surface plot)
    # Using thermal resistance network with convection and conduction
    # Create 2D arrays: wall_temp_3d[i, j] = temperature at mdot[i] and wall_distance[j]
    wall_temp_3d = np.zeros((len(mdot), wall_samples))

    for i in range(len(mdot)):
        T_gas = stag_temp[i]  # Hot gas temperature (stagnation)
        T_external_fluid = external_temp  # External coolant temperature
        h_gas = h_gas_array[i]  # Use calculated heat transfer coefficient

        # Calculate thermal resistances per unit length (cylindrical)
        # R_conv_inner = 1 / (h_gas * 2*pi*r_inner)
        # R_cond = ln(r_outer/r_inner) / (2*pi*k)
        # R_conv_outer = 1 / (h_external * 2*pi*r_outer)
        R_conv_inner = 1.0 / (h_gas * 2 * np.pi * r_inner_wall)
        R_cond = np.log(r_outer_wall / r_inner_wall) / (2 * np.pi * wall_conductivity)
        R_conv_outer = 1.0 / (h_external * 2 * np.pi * r_outer_wall)

        R_total = R_conv_inner + R_cond + R_conv_outer

        # Heat transfer per unit length
        q_per_length = (T_gas - T_external_fluid) / R_total  # W/m

        # Calculate temperatures at each radial position
        # T_inner_surface = T_gas - q' * R_conv_inner
        T_inner_surface = T_gas - q_per_length * R_conv_inner

        # Temperature distribution through the wall (conduction part)
        # T(r) = T_inner_surface - (q' / (2*pi*k)) * ln(r / r_inner)
        wall_temp_3d[i, :] = T_inner_surface - (q_per_length / (2 * np.pi * wall_conductivity)) * np.log(r_wall / r_inner_wall)

    # Create figure with all traces, control visibility via dropdown
    fig = go.Figure()

    # Add all traces - only first one visible initially
    # Trace 0: Stagnation Temperature
    fig.add_trace(
        go.Scatter(
            x=mdot,
            y=stag_temp,
            mode='lines+markers',
            name='Stagnation Temperature',
            line=dict(width=3),
            visible=True
        )
    )

    # Trace 1: Stagnation Pressure
    fig.add_trace(
        go.Scatter(
            x=mdot,
            y=stag_pressure,
            mode='lines+markers',
            name='Stagnation Pressure',
            line=dict(width=3),
            visible=False
        )
    )

    # Trace 2: Stagnation Density
    fig.add_trace(
        go.Scatter(
            x=mdot,
            y=stag_density,
            mode='lines+markers',
            name='Stagnation Density',
            line=dict(width=3),
            visible=False
        )
    )

    # Trace 3: Exit Velocity
    fig.add_trace(
        go.Scatter(
            x=mdot,
            y=exit_velocity,
            mode='lines+markers',
            name='Exit Velocity',
            line=dict(width=3),
            visible=False
        )
    )

    # Trace 4: Mach Number
    fig.add_trace(
        go.Scatter(
            x=mdot,
            y=mach,
            mode='lines+markers',
            name='Mach Number',
            line=dict(width=3),
            visible=False
        )
    )

    # Trace 5: Wall Temperature (3D Surface)
    fig.add_trace(
        go.Surface(
            x=wall_distance,
            y=mdot,
            z=wall_temp_3d,
            colorscale='Jet',
            colorbar=dict(title="Temperature (K)"),
            visible=False
        )
    )

    # Create dropdown menu to select which plot to show
    buttons = [
        dict(
            label="Stagnation Temperature",
            method="update",
            args=[
                {"visible": [True, False, False, False, False, False]},
                {"title": {"text": "Stagnation Temperature vs Mass Flow Rate"},
                 "xaxis": {"title": {"text": "Mass Flow Rate (kg/s)"}},
                 "yaxis": {"title": {"text": "Temperature (K)"}}}
            ]
        ),
        dict(
            label="Stagnation Pressure",
            method="update",
            args=[
                {"visible": [False, True, False, False, False, False]},
                {"title": {"text": "Stagnation Pressure vs Mass Flow Rate"},
                 "xaxis": {"title": {"text": "Mass Flow Rate (kg/s)"}},
                 "yaxis": {"title": {"text": "Pressure (Pa)"}}}
            ]
        ),
        dict(
            label="Stagnation Density",
            method="update",
            args=[
                {"visible": [False, False, True, False, False, False]},
                {"title": {"text": "Stagnation Density vs Mass Flow Rate"},
                 "xaxis": {"title": {"text": "Mass Flow Rate (kg/s)"}},
                 "yaxis": {"title": {"text": "Density (kg/m³)"}}}
            ]
        ),
        dict(
            label="Exit Velocity",
            method="update",
            args=[
                {"visible": [False, False, False, True, False, False]},
                {"title": {"text": "Exit Velocity vs Mass Flow Rate"},
                 "xaxis": {"title": {"text": "Mass Flow Rate (kg/s)"}},
                 "yaxis": {"title": {"text": "Velocity (m/s)"}}}
            ]
        ),
        dict(
            label="Mach Number",
            method="update",
            args=[
                {"visible": [False, False, False, False, True, False]},
                {"title": {"text": "Mach Number vs Mass Flow Rate"},
                 "xaxis": {"title": {"text": "Mass Flow Rate (kg/s)"}},
                 "yaxis": {"title": {"text": "Mach Number"}}}
            ]
        ),
        dict(
            label="Wall Temperature (3D)",
            method="update",
            args=[
                {"visible": [False, False, False, False, False, True]},
                {"title": {"text": "Wall Temperature Distribution"},
                 "scene": {
                     "xaxis": {"title": {"text": "Distance through Wall (mm)"}},
                     "yaxis": {"title": {"text": "Mass Flow Rate (kg/s)"}},
                     "zaxis": {"title": {"text": "Temperature (K)"}},
                     "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.3}}
                 }}
            ]
        )
    ]

    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        title_text="Stagnation Temperature vs Mass Flow Rate",
        xaxis=dict(title="Mass Flow Rate (kg/s)"),
        yaxis=dict(title="Temperature (K)"),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.1,
            yanchor="top"
        )]
    )

    # Add annotation if solutions were excluded due to divergence
    if failure_index is not None:
        fig.add_annotation(
            text="Note: Supersonic solutions are not supported and were excluded",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255, 255, 255, 0.8)"
        )

    # Convert to HTML string
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    return html

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Error: Expected JSON input data as argument")
        sys.exit(1)
    print(main(sys.argv[1]))