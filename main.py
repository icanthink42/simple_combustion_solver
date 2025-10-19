from flow_state import FlowState
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys


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
        radius = float(params['radius'])  # m

        # Optional parameters with defaults
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
    area = np.pi * radius**2

    stag_temp = np.zeros(len(mdot))
    stag_pressure = np.zeros(len(mdot))
    stag_density = np.zeros(len(mdot))
    exit_velocity = np.zeros(len(mdot))
    mach = np.zeros(len(mdot))

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

    if failure_index is not None:
        mdot = mdot[:failure_index]
        stag_temp = stag_temp[:failure_index]
        stag_pressure = stag_pressure[:failure_index]
        stag_density = stag_density[:failure_index]
        exit_velocity = exit_velocity[:failure_index]
        mach = mach[:failure_index]

    # Create a 2x2 subplot grid
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Stagnation Temperature vs Velocity',
            'Stagnation Pressure vs Velocity',
            'Stagnation Density vs Velocity',
            'Exit Velocity vs Velocity',
            'Mach vs Velocity'
        )
    )

    # Add traces to each subplot
    fig.add_trace(
        go.Scatter(x=mdot, y=stag_temp, mode='lines', name='Temperature'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=mdot, y=stag_pressure, mode='lines', name='Pressure'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=mdot, y=stag_density, mode='lines', name='Density'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=mdot, y=exit_velocity, mode='lines', name='Velocity'),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=mdot, y=mach, mode='lines', name='Mach'),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Flow Properties vs Velocity"
    )

    # Add annotation if solutions were excluded due to divergence
    if failure_index is not None:
        fig.add_annotation(
            text="Note: Supersonic solutions are not supported and were excluded",
            xref="paper", yref="paper",
            x=0.5, y=1.05,  # Position above the plots
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255, 255, 255, 0.8)"
        )

    # Update x and y axis labels
    fig.update_xaxes(title_text="Mass Flow Rate (kg/s)")
    fig.update_yaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (Pa)", row=1, col=2)
    fig.update_yaxes(title_text="Density (kg/m³)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="Mach", row=2, col=3)

    # Convert to HTML string
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    return html

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Error: Expected JSON input data as argument")
        sys.exit(1)
    print(main(sys.argv[1]))