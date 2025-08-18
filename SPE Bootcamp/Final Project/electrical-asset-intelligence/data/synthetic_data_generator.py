import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_transformer_data(num_samples=1000):
    """Generate synthetic transformer data based on real-world experience at Bapco Gas"""
    np.random.seed(42)
    
    # Baseline healthy values (based on typical transformer specs)
    baseline_temp = 65  # °C
    baseline_winding = 70  # °C
    baseline_load = 75  # %
    baseline_insulation = 1500  # MΩ
    baseline_pd = 50  # pC
    
    # Create time series
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    
    # Initialize arrays
    oil_temps = []
    winding_temps = []
    loads = []
    insulation_resistances = []
    partial_discharges = []
    oil_quality = []
    maintenance_flags = []
    health_status = []
    failure_modes = []
    
    # Generate realistic patterns with failures
    for i in range(num_samples):
        # Normal operating conditions with some variation
        oil_temp = baseline_temp + np.random.normal(0, 5)
        winding_temp = baseline_winding + np.random.normal(0, 6)
        load = baseline_load + np.random.normal(0, 10)
        insulation = baseline_insulation * np.random.normal(1, 0.05)
        pd_level = baseline_pd + np.random.normal(0, 10)
        
        # Introduce some degradation patterns
        if i > 200 and i < 300:  # Gradual insulation degradation
            insulation *= (1 - (i-200)/100 * 0.3)
            pd_level += (i-200)/10 * 5
            
        if i > 500 and i < 550:  # Overheating event
            oil_temp += (i-500)/5 * 8
            winding_temp += (i-500)/5 * 10
            
        # Add random maintenance events
        maintenance_flag = 0
        if random.random() < 0.02:  # 2% chance of maintenance on any day
            maintenance_flag = 1
            # After maintenance, reset some parameters
            insulation = baseline_insulation * np.random.normal(1, 0.02)
            oil_temp = baseline_temp + np.random.normal(0, 2)
            
        # Determine health status and failure mode
        failure_mode = "Normal"
        if insulation < 800 or pd_level > 200 or (oil_temp > 90 and winding_temp > 95):
            status = "Critical"
            if insulation < 800:
                failure_mode = "Insulation degradation"
            elif pd_level > 200:
                failure_mode = "Partial discharge"
            else:
                failure_mode = "Overheating"
        elif insulation < 1000 or pd_level > 150 or (oil_temp > 85 and winding_temp > 90):
            status = "Warning"
            if insulation < 1000:
                failure_mode = "Early insulation degradation"
            elif pd_level > 150:
                failure_mode = "Elevated partial discharge"
            else:
                failure_mode = "Temperature anomaly"
        else:
            status = "Healthy"
            
        # Add to lists
        oil_temps.append(max(40, min(120, oil_temp)))  # Clamp to realistic range
        winding_temps.append(max(45, min(125, winding_temp)))
        loads.append(max(20, min(100, load)))
        insulation_resistances.append(max(500, min(2000, insulation)))
        partial_discharges.append(max(20, min(300, pd_level)))
        oil_quality.append(random.choice(["Good", "Fair", "Poor"]))
        maintenance_flags.append(maintenance_flag)
        health_status.append(status)
        failure_modes.append(failure_mode)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'oil_temp_c': oil_temps,
        'winding_temp_c': winding_temps,
        'load_percent': loads,
        'insulation_resistance_mohm': insulation_resistances,
        'partial_discharge_pc': partial_discharges,
        'oil_quality': oil_quality,
        'maintenance_performed': maintenance_flags,
        'health_status': health_status,
        'failure_mode': failure_modes
    })
    
    return df

def generate_motor_data(num_samples=1000):
    """Generate synthetic motor data based on experience with motors at Bapco Gas"""
    np.random.seed(42)
    
    # Baseline healthy values
    baseline_vibration = 1.5  # mm/s
    baseline_current = 100  # A
    baseline_voltage = 415  # V
    baseline_pf = 0.85  # Power factor
    baseline_temp = 60  # °C
    
    # Create time series
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    
    # Initialize arrays
    vibrations = []
    currents = []
    voltages = []
    power_factors = []
    temperatures = []
    runtime_hours = []
    bearing_conditions = []
    health_status = []
    failure_modes = []
    
    # Generate realistic patterns with failures
    runtime = 0
    for i in range(num_samples):
        # Daily runtime (8-24 hours)
        daily_runtime = random.randint(8, 24)
        runtime += daily_runtime
        
        # Normal operating conditions with some variation
        vibration = baseline_vibration + np.random.normal(0, 0.5)
        current = baseline_current + np.random.normal(0, 10)
        voltage = baseline_voltage + np.random.normal(0, 5)
        pf = max(0.7, min(0.95, baseline_pf + np.random.normal(0, 0.05)))
        temp = baseline_temp + np.random.normal(0, 5)
        
        # Introduce bearing degradation pattern
        if i > 300 and i < 400:  # Bearing wear developing
            vibration += (i-300)/10 * 0.8
            temp += (i-300)/10 * 1.5
            
        # Introduce electrical fault pattern
        if i > 700 and i < 750:  # Electrical issue
            current += (i-700)/5 * 8
            pf -= (i-700)/50 * 0.1
            
        # Determine health status and failure mode
        failure_mode = "Normal"
        if vibration > 4.5 or (current > 130 and temp > 80):
            status = "Critical"
            if vibration > 4.5:
                failure_mode = "Bearing failure"
            else:
                failure_mode = "Electrical fault"
        elif vibration > 3.0 or (current > 120 and temp > 75):
            status = "Warning"
            if vibration > 3.0:
                failure_mode = "Early bearing wear"
            else:
                failure_mode = "Electrical anomaly"
        else:
            status = "Healthy"
            
        # Add to lists
        vibrations.append(max(0.5, min(10, vibration)))
        currents.append(max(70, min(150, current)))
        voltages.append(max(380, min(440, voltage)))
        power_factors.append(pf)
        temperatures.append(max(40, min(100, temp)))
        runtime_hours.append(runtime)
        bearing_conditions.append(random.choice(["Good", "Fair", "Worn"]))
        health_status.append(status)
        failure_modes.append(failure_mode)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'vibration_mm_s': vibrations,
        'current_a': currents,
        'voltage_v': voltages,
        'power_factor': power_factors,
        'temperature_c': temperatures,
        'runtime_hours': runtime_hours,
        'bearing_condition': bearing_conditions,
        'health_status': health_status,
        'failure_mode': failure_modes
    })
    
    return df

def generate_capacitor_bank_data(num_samples=1000):
    """Generate synthetic capacitor bank data based on Week 6 experience"""
    np.random.seed(42)
    
    # Baseline healthy values
    baseline_voltage = 11  # kV
    baseline_current = 50  # A
    baseline_pf = 0.95  # Power factor
    baseline_harmonics = 3  # %
    baseline_temp = 45  # °C
    baseline_discharge = 10  # s
    
    # Create time series
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    
    # Initialize arrays
    voltages = []
    currents = []
    power_factors = []
    harmonics = []
    temperatures = []
    discharge_times = []
    fuse_statuses = []
    health_status = []
    failure_modes = []
    
    # Generate realistic patterns with failures
    for i in range(num_samples):
        # Normal operating conditions with some variation
        voltage = baseline_voltage + np.random.normal(0, 0.5)
        current = baseline_current + np.random.normal(0, 5)
        pf = max(0.85, min(0.98, baseline_pf + np.random.normal(0, 0.02)))
        harmonic = max(2, min(10, baseline_harmonics + np.random.normal(0, 1)))
        temp = baseline_temp + np.random.normal(0, 3)
        discharge = baseline_discharge + np.random.normal(0, 0.5)
        
        # Introduce capacitor degradation pattern
        if i > 400 and i < 450:  # Capacitor degradation
            pf -= (i-400)/50 * 0.05
            harmonic += (i-400)/5 * 0.8
            temp += (i-400)/5 * 0.5
            
        # Introduce fuse failure pattern
        if i > 800 and i < 820:  # Fuse failure event
            current = max(20, current - 15)
            pf -= 0.1
            discharge += 2
            
        # Determine health status and failure mode
        failure_mode = "Normal"
        if harmonic > 8 or (temp > 60 and pf < 0.88):
            status = "Critical"
            if harmonic > 8:
                failure_mode = "Harmonic overload"
            else:
                failure_mode = "Thermal failure"
        elif harmonic > 6 or (temp > 55 and pf < 0.90):
            status = "Warning"
            if harmonic > 6:
                failure_mode = "Harmonic distortion"
            else:
                failure_mode = "Temperature rise"
        else:
            status = "Healthy"
            
        # Add to lists
        voltages.append(max(10, min(12, voltage)))
        currents.append(max(30, min(60, current)))
        power_factors.append(pf)
        harmonics.append(harmonic)
        temperatures.append(max(35, min(70, temp)))
        discharge_times.append(max(8, min(15, discharge)))
        fuse_statuses.append("Good" if i != 810 else "Blown")
        health_status.append(status)
        failure_modes.append(failure_mode)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'voltage_kv': voltages,
        'current_a': currents,
        'power_factor': power_factors,
        'harmonic_distortion_percent': harmonics,
        'temperature_c': temperatures,
        'discharge_time_s': discharge_times,
        'fuse_status': fuse_statuses,
        'health_status': health_status,
        'failure_mode': failure_modes
    })
    
    return df

def generate_ups_data(num_samples=1000):
    """Generate synthetic UPS data based on experience with battery checks"""
    np.random.seed(42)
    
    # Baseline healthy values
    baseline_voltage = 2.1  # V per cell
    baseline_temp = 25  # °C
    baseline_load = 60  # %
    baseline_backup = 30  # minutes
    baseline_efficiency = 92  # %
    
    # Create time series
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    
    # Initialize arrays
    voltages = []
    temperatures = []
    loads = []
    backup_times = []
    efficiencies = []
    operating_modes = []
    health_status = []
    failure_modes = []
    
    # Generate realistic patterns with failures
    for i in range(num_samples):
        # Normal operating conditions with some variation
        voltage = baseline_voltage + np.random.normal(0, 0.05)
        temp = baseline_temp + np.random.normal(0, 2)
        load = baseline_load + np.random.normal(0, 10)
        backup = baseline_backup + np.random.normal(0, 5)
        efficiency = baseline_efficiency + np.random.normal(0, 1)
        
        # Introduce battery degradation pattern
        if i > 250 and i < 300:  # Battery aging
            voltage -= (i-250)/50 * 0.05
            backup -= (i-250)/5 * 1
            temp += (i-250)/10 * 0.5
            
        # Introduce charger issue pattern
        if i > 600 and i < 630:  # Charger problem
            voltage -= (i-600)/30 * 0.15
            efficiency -= (i-600)/3 * 0.5
            
        # Determine health status and failure mode
        failure_mode = "Normal"
        if voltage < 1.9 or backup < 15 or (temp > 40 and efficiency < 85):
            status = "Critical"
            if voltage < 1.9:
                failure_mode = "Battery failure"
            elif backup < 15:
                failure_mode = "Capacity loss"
            else:
                failure_mode = "Charger issue"
        elif voltage < 2.0 or backup < 20 or (temp > 35 and efficiency < 88):
            status = "Warning"
            if voltage < 2.0:
                failure_mode = "Battery degradation"
            elif backup < 20:
                failure_mode = "Reduced capacity"
            else:
                failure_mode = "Charger anomaly"
        else:
            status = "Healthy"
            
        # Add to lists
        voltages.append(max(1.8, min(2.2, voltage)))
        temperatures.append(max(20, min(45, temp)))
        loads.append(max(20, min(100, load)))
        backup_times.append(max(5, min(40, backup)))
        efficiencies.append(max(80, min(95, efficiency)))
        operating_modes.append(random.choice(["Float", "Boost", "Inverter"]))
        health_status.append(status)
        failure_modes.append(failure_mode)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'voltage_per_cell_v': voltages,
        'temperature_c': temperatures,
        'load_percent': loads,
        'backup_time_min': backup_times,
        'charger_efficiency_percent': efficiencies,
        'operating_mode': operating_modes,
        'health_status': health_status,
        'failure_mode': failure_modes
    })
    
    return df

# Generate and save all datasets
if __name__ == "__main__":
    import random
    
    print("Generating synthetic datasets...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Generate and save transformer data
    transformer_df = generate_transformer_data()
    transformer_df.to_csv('data/transformer_data.csv', index=False)
    print(f"✅ Transformer data generated ({len(transformer_df)} records)")
    
    # Generate and save motor data
    motor_df = generate_motor_data()
    motor_df.to_csv('data/motor_data.csv', index=False)
    print(f"✅ Motor data generated ({len(motor_df)} records)")
    
    # Generate and save capacitor bank data
    capacitor_df = generate_capacitor_bank_data()
    capacitor_df.to_csv('data/capacitor_data.csv', index=False)
    print(f"✅ Capacitor bank data generated ({len(capacitor_df)} records)")
    
    # Generate and save UPS data
    ups_df = generate_ups_data()
    ups_df.to_csv('data/ups_data.csv', index=False)
    print(f"✅ UPS data generated ({len(ups_df)} records)")
    
    print("\nAll datasets generated successfully!")
    print("You can now proceed to data preprocessing and model development.")