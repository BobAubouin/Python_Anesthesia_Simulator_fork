from python_anesthesia_simulator.pk_models import AtracuriumModel
from python_anesthesia_simulator.pd_models import NMB_model
from python_anesthesia_simulator.patient import Patient
import numpy as np
import matplotlib.pyplot as plt

# %% Test1: Bolus
# Simulate the response to an initial atracurium bolus of 500 ug/ml
# The bolus is administered directly in the central compartment and it is
# simulated by initializing the states.
# Since the states are concentrations defined in ug/ml the initial bolus dose
# is converted accordinly by dividing it to the volume of the central
# compartment.

# These are not affecting the Ward Weatherley Lago model
age = 35
height = 170
sex = 0

# Parameters of the Ward Weatherley Lago test model 1
weight_test_model_default = 75
V1_test_model_default = 49

# State initialization to simulate the atracurium bolus administration
x0 = np.array([500/V1_test_model_default, 0, 0, 0])

# Create dictionaries of models parameters that matches the default to check
# that the two initialization methodologies give the same results
pk_parameters_custom = {
    'V1': 49.0,          # Volume of the central compartment [ml/kg]
    'V2': 157.0,         # Volume of the peripheral compartment [ml/kg]
    'Cl': 5.5,           # Clearance [ml/min/kg]
    't12_alpha': 2.06,   # First half-life time [min]
    't12_beta': 19.9,    # Second half-life time [min]
    'ke0': 0.1,          # Transfer rate of the first effect-site compartment [1/min]
    'tau': 6.2670        # Time constant of the second effect-site compartment [min]
}
pd_parameters_custom = {
    'C50': 0.625,          # Half effect concentration [ug/ml]
    'gamma': 4.25          # Slope
}

# Create Atracurium PK model objects
test_model_default = AtracuriumModel([age, height, weight_test_model_default, sex])
test_model_custom = AtracuriumModel([age, height, weight_test_model_default, sex],
                                    model_params=pk_parameters_custom)
test_model_default_one_step = AtracuriumModel([age, height, weight_test_model_default, sex])
test_model_custom_one_step = AtracuriumModel([age, height, weight_test_model_default, sex],
                                             model_params=pk_parameters_custom)
# Create Atracurium PD model objects
test_hill_default = NMB_model()
test_hill_custom = NMB_model(hill_param=pd_parameters_custom)
# test_hill_custom.plot_surface()

# Create Patient objects that implements the atracurium models
George_1 = Patient([age, height, weight_test_model_default, sex])
George_one_step = Patient([age, height, weight_test_model_default, sex])

# Simulation parameters
Tsim = 60*100
ts = 1
Nsim = int(Tsim/ts)
atracurium_infusion_profile = np.zeros((Nsim,))        # ug/s
time = np.arange(Tsim)  # Time axis

# Simulation by using full_sim
# Simulate the free response to a bolus administration with default initialization
x_default = test_model_default.full_sim(u=atracurium_infusion_profile, x0=x0)
nmb_default = test_hill_default.compute_nmb(x_default[3])
# Find response peak and its timing
peak_idx_default = np.argmin(nmb_default)
min_nmb_default = nmb_default[peak_idx_default]
peak_time_default = time[peak_idx_default]
# Find  recovery time NMB>= 50
recovery_idx_default = np.where(nmb_default <= 50)[0][-1]
recovery_time_default = time[recovery_idx_default]

# Simulate the free response to a bolus administration with custom initialization
x_custom = test_model_custom.full_sim(u=atracurium_infusion_profile, x0=x0)
nmb_custom = test_hill_custom.compute_nmb(x_custom[3])
# Find response peak and its timing
peak_idx_custom = np.argmin(nmb_custom)
min_nmb_custom = nmb_custom[peak_idx_custom]
peak_time_custom = time[peak_idx_custom]
# Find  recovery time NMB>= 50
recovery_idx_custom = np.where(nmb_custom <= 50)[0][-1]
recovery_time_custom = time[recovery_idx_custom]

# Simulate the free response to a bolus administration with the Patient object
df_george_1 = George_1.full_sim(u_atra=atracurium_infusion_profile,
                                x0_atra=x0)
nmb_george_1 = df_george_1.NMB
# Find response peak and its timing
peak_idx_george_1 = np.argmin(nmb_george_1)
min_nmb_george_1 = nmb_custom[peak_idx_george_1]
peak_time_george_1 = time[peak_idx_george_1]
# Find  recovery time NMB>= 50
recovery_idx_george_1 = np.where(nmb_george_1 <= 50)[0][-1]
recovery_time_george_1 = time[recovery_idx_george_1]

Ce_default_one_step = np.zeros((Nsim,))
nmb_default_one_step = np.zeros((Nsim,))
test_model_default_one_step.initialize_state(x0)
# Simulation by using one_step
for k in range(Nsim-1):
    uAtra_k = atracurium_infusion_profile[k]
    Ce_default_one_step[k] = test_model_default_one_step.one_step(uAtra_k)
    nmb_default_one_step[k] = test_hill_default.compute_nmb(Ce_default_one_step[k])
# Find response peak and its timing
peak_idx_default_one_step = np.argmin(nmb_default_one_step)
min_nmb_default_one_step = nmb_default_one_step[peak_idx_default_one_step]
peak_time_default_one_step = time[peak_idx_default_one_step]
# Find  recovery time NMB>= 50
recovery_idx_default_one_step = np.where(nmb_default_one_step <= 50)[0][-2]
recovery_time_default_one_step = time[recovery_idx_default_one_step]

Ce_custom_one_step = np.zeros((Nsim,))
nmb_custom_one_step = np.zeros((Nsim,))
test_model_custom_one_step.initialize_state(x0)
# Simulation by using one_step
for k in range(Nsim-1):
    uAtra_k = atracurium_infusion_profile[k]
    Ce_custom_one_step[k] = test_model_custom_one_step.one_step(uAtra_k)
    nmb_custom_one_step[k] = test_hill_custom.compute_nmb(Ce_custom_one_step[k])
# Find response peak and its timing
peak_idx_custom_one_step = np.argmin(nmb_custom_one_step)
min_nmb_custom_one_step = nmb_custom_one_step[peak_idx_custom_one_step]
peak_time_custom_one_step = time[peak_idx_custom_one_step]
# Find  recovery time NMB>= 50
recovery_idx_custom_one_step = np.where(nmb_custom_one_step <= 50)[0][-2]
recovery_time_custom_one_step = time[recovery_idx_custom_one_step]

George_one_step.initialized_at_given_state(x0_atra=x0)
for k in range(Nsim-1):
    uAtra_k = atracurium_infusion_profile[k]
    George_one_step.one_step(u_atra=uAtra_k,
                             noise=False)
nmb_george_one_step = George_one_step.dataframe['NMB']
# Find response peak and its timing
peak_idx_george_one_step = np.argmin(nmb_george_one_step)
min_nmb_george_one_step = nmb_george_one_step[peak_idx_george_one_step]
peak_time_george_one_step = time[peak_idx_george_one_step]
# Find  recovery time NMB>= 50
recovery_idx_george_one_step = np.where(nmb_george_one_step <= 50)[0][-1]
recovery_time_george_one_step = time[recovery_idx_george_one_step]

# %% Test 2: Infusion simulation
# These are not affecting the Ward Weatherley Lago model
age_test_2 = 35
height_test_2 = 170
sex_test_2 = 0

# Parameters of the Ward Weatherley Lago test model 1
weight_test_2 = 75
V1_test_2 = 49


# Create dictionaries of models parameters that matches the default to check
# that the two initialization methodologies give the same results
pk_parameters_custom_test_2 = {
    'V1': 49.0,          # Volume of the central compartment [ml/kg]
    'V2': 157.0,         # Volume of the peripheral compartment [ml/kg]
    'Cl': 5.5,           # Clearance [ml/min/kg]
    't12_alpha': 2.06,   # First half-life time [min]
    't12_beta': 19.9,    # Second half-life time [min]
    'ke0': 0.1,          # Transfer rate of the first effect-site compartment [1/min]
    'tau': 6.2670        # Time constant of the second effect-site compartment [min]
}
pd_parameters_custom_test_2 = {
    'C50': 0.625,          # Half effect concentration [ug/ml]
    'gamma': 4.25          # Slope
}

# Create Atracurium PK model objects
test_model_default_test_2 = AtracuriumModel([age_test_2, height_test_2, weight_test_2, sex_test_2])
test_model_custom_test_2 = AtracuriumModel([age_test_2, height_test_2, weight_test_2, sex_test_2],
                                           model_params=pk_parameters_custom_test_2)
# Create Atracurium PD model objects
test_hill_default_test_2 = NMB_model()
test_hill_custom_test_2 = NMB_model(hill_param=pd_parameters_custom_test_2)
# test_hill_custom.plot_surface()

# Create Patient objects that implements the atracurium models
George_test_2 = Patient([age_test_2, height_test_2, weight_test_2, sex_test_2])

# Simulation parameters
Tsim_test_2 = 60*100
ts_test_2 = 1
Nsim_test_2 = int(Tsim_test_2/ts_test_2)
atracurium_infusion_profile_test_2 = np.zeros((Nsim_test_2,))        # ug/s
time_test_2 = np.arange(Tsim_test_2)  # Time axis


# Simulation parameters
atracurium_infusion_profile_test_2 = np.zeros((Nsim_test_2,))  # ug/s
time_test_2 = np.arange(Tsim_test_2)  # Time axis
# Atracurium infusion profile
atracurium_infusion_profile_test_2[0:int(50/ts_test_2)] = 20  # 20 ug/s for 50 seconds
atracurium_infusion_profile_test_2[int(150/ts_test_2):] = 8   # 8 ug/s from 150s onward

# Simulate the forced response to infusion with default initialization
x_default_test_2 = test_model_default_test_2.full_sim(u=atracurium_infusion_profile_test_2)
nmb_default_test_2 = test_hill_default_test_2.compute_nmb(x_default_test_2[3])
# Find  induction time NMB<= 20
induction_idx_default_test_2 = np.where(nmb_default_test_2 <= 20)[0][0]
induction_time_default_test_2 = time[induction_idx_default_test_2]

# Simulate the forced response to infusion with custom initialization
x_custom_test_2 = test_model_custom_test_2.full_sim(u=atracurium_infusion_profile_test_2)
nmb_custom_test_2 = test_hill_custom_test_2.compute_nmb(x_custom_test_2[3])
# Find  induction time NMB<= 20
induction_idx_custom_test_2 = np.where(nmb_custom_test_2 <= 20)[0][0]
induction_time_custom_test_2 = time[induction_idx_custom_test_2]

# Simulate the free response to a bolus administration with the Patient object
df_george_test_2 = George_test_2.full_sim(u_atra=atracurium_infusion_profile_test_2)
nmb_george_test_2 = df_george_test_2.NMB
# Find  induction time NMB<= 20
induction_idx_george_2 = np.where(nmb_george_test_2 <= 20)[0][0]
induction_time_george_2 = time[induction_idx_george_2]


# %% Tests
def test_default_initialization_atracurium():
    """Ensure that the default models give correct results"""
    # Check the Weatherley Hill curve at relevant atracurium concentrations
    assert test_hill_default.compute_nmb(0) == 100
    assert test_hill_default.compute_nmb(0.4) >= 80
    assert test_hill_default.compute_nmb(0.6) >= 50
    assert test_hill_default.compute_nmb(0.8) <= 30
    assert test_hill_default.compute_nmb(1) <= 20
    assert test_hill_default.compute_nmb(2) <= 10

    assert test_hill_custom.compute_nmb(0) == 100
    assert test_hill_custom.compute_nmb(0.4) >= 80
    assert test_hill_custom.compute_nmb(0.6) >= 50
    assert test_hill_custom.compute_nmb(0.8) <= 30
    assert test_hill_custom.compute_nmb(1) <= 20
    assert test_hill_custom.compute_nmb(2) <= 10

    # Bolus response
    # Check the key characteristic of the average patient response
    assert min_nmb_default < 1
    assert peak_time_default > 500
    assert 3000 <= recovery_time_default <= 4000

    assert min_nmb_custom < 1
    assert peak_time_custom > 500
    assert 3000 <= recovery_time_custom <= 4000

    assert min_nmb_george_1 < 1
    assert peak_time_george_1 > 500
    assert 3000 <= recovery_time_george_1 <= 4000

    assert min_nmb_default_one_step < 1
    assert peak_time_default_one_step > 500
    assert 3000 <= recovery_time_default_one_step <= 4000

    assert min_nmb_custom_one_step < 1
    assert peak_time_custom_one_step > 500
    assert 3000 <= recovery_time_custom_one_step <= 4000

    assert min_nmb_george_one_step < 1
    assert peak_time_george_one_step > 500
    assert 3000 <= recovery_time_george_one_step <= 4000

    # Infusion response
    # Check the key characteristic of the average patient response
    assert 2000 <= induction_time_default_test_2 <= 4000
    assert 2000 <= induction_time_custom_test_2 <= 4000
    assert 2000 <= induction_time_george_2 <= 4000


# %% Plots
if __name__ == '__main__':

    # Plot bolus responses
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(time, x_default[i], label=f'x{i+1}(t)')
    plt.plot(df_george_1.Time, df_george_1.x_atra_1, '--')
    plt.plot(df_george_1.Time, df_george_1.x_atra_2, '--')
    plt.plot(df_george_1.Time, df_george_1.x_atra_3, '--')
    plt.plot(df_george_1.Time, df_george_1.x_atra_4, '--')
    plt.plot(George_one_step.dataframe['Time'], George_one_step.dataframe['x_atra_1'], '--')
    plt.plot(George_one_step.dataframe['Time'], George_one_step.dataframe['x_atra_2'], '--')
    plt.plot(George_one_step.dataframe['Time'], George_one_step.dataframe['x_atra_3'], '--')
    plt.plot(George_one_step.dataframe['Time'], George_one_step.dataframe['x_atra_4'], '--')
    plt.xlabel('Time (s)')
    plt.ylabel('State Value')
    plt.title('Evolution of States Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    # plt.plot(time, nmb_default, 'k-', linewidth=1, label='NMB')
    # plt.plot(df_george_1.Time, df_george_1.NMB, 'g--', linewidth=1)
    plt.plot(time, nmb_default, 'k-', label='NMB')
    plt.plot(df_george_1.Time, df_george_1.NMB, 'g--')
    plt.plot(time, nmb_default_one_step, 'm--')
    # Set axes limits
    plt.xlim(0, 6000)  # X-axis from 0-6000 seconds
    plt.ylim(0, 120)   # Y-axis from 0-120%
    # Style
    plt.xlabel('time (s)', fontsize=12)
    plt.ylabel('NMB (%)', fontsize=12)
    plt.title('Average patient', fontsize=14, pad=20)
    # Grid and ticks
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, 7000, 1000))  # Major ticks every 1000s
    plt.yticks(np.arange(0, 140, 20))     # Major ticks every 20%
    plt.tight_layout()
    plt.show()

    # Plot infusion responses
    # Create figure with two subplots
    plt.figure(figsize=(10, 8))
    # First subplot - NMB
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    plt.plot(time_test_2, nmb_default_test_2, 'k-', label='NMB')
    plt.plot(df_george_test_2.Time, df_george_test_2.NMB, 'g--')
    # Set axes limits
    plt.xlim(0, 6000)  # X-axis from 0-6000 seconds
    plt.ylim(0, 120)   # Y-axis from 0-120%
    # Style
    plt.ylabel('NMB (%)', fontsize=12)
    plt.title('Average patient', fontsize=14, pad=20)
    # Grid and ticks
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, 7000, 1000))  # Major ticks every 1000s
    plt.yticks(np.arange(0, 140, 20))     # Major ticks every 20%
    # Second subplot - Infusion rate
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.plot(time_test_2, atracurium_infusion_profile_test_2, 'k-', label='Infusion Rate')
    # Set axes limits
    plt.xlim(0, 6000)
    plt.ylim(0, 25)  # Adjust based on your infusion rate range
    # Style
    plt.xlabel('time (s)', fontsize=12)
    plt.ylabel('Infusion Rate (ug/s)', fontsize=12)
    # Grid and ticks
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, 7000, 1000))
    plt.yticks(np.arange(0, 30, 5))  # Adjust ticks based on your infusion rate
    plt.tight_layout()
    plt.show()
