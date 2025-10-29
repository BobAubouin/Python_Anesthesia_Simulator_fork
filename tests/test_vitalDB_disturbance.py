import matplotlib.pyplot as plt
import numpy as np

from python_anesthesia_simulator import patient, TCIController

# %% Simulation setup
# Simulation duration in seconds
Tsim = 90 * 60
# Patient physical data
age = 20
weight = 70
height = 170
gender = 0
# Sampling time
ts = 1

Nsim = int(Tsim / ts)

George = patient.Patient(
    [age, height, weight, gender],
    ts=ts,
    model_propo="Eleveld",
    model_remi="Eleveld",
    model_stimuli='VitalDB',
)

# Initialize tci
tci_propo = TCIController(
    [age, height, weight, gender],
    drug_name="Propofol",
    drug_concentration=10,
    sampling_time=ts,
    model_used="Eleveld",
)
tci_remi = TCIController(
    [age, height, weight, gender],
    drug_name="Remifentanil",
    drug_concentration=50,
    sampling_time=ts,
    model_used="Eleveld",
)

propo_target = 4
remi_target = 3

for k in range(Nsim - 1):
    uProp_k = tci_propo.one_step(propo_target) / 3600 * 10  # convert to mg/s
    uRemi_k = tci_remi.one_step(remi_target) / 3600 * 50    # convert to ug/s
    George.one_step(u_propo=uProp_k, u_remi=uRemi_k, noise=False)

if __name__ == '__main__':

    fig, ax = plt.subplots(4)
    ax[0].plot(George.dataframe['Time'] / 60, George.dataframe['BIS'])
    ax[0].set_ylabel('BIS')
    ax[0].grid()
    ax[1].plot(George.dataframe['Time'] / 60, George.dataframe['HR'])
    ax[1].set_ylabel('HR (bpm)')
    ax[1].grid()
    ax[2].plot(George.dataframe['Time'] / 60, George.dataframe['SAP'])
    ax[2].set_ylabel('SAP (mmHg)')
    ax[2].grid()
    ax[3].plot(George.dataframe['Time'] / 60, George.dataframe['DAP'])
    ax[3].set_ylabel('DAP (mmHg)')
    ax[3].set_xlabel('Time (min)')
    ax[3].grid()
    for i in range(4):
        ax[i].axvline(x=3, color='g', linestyle='--')
        ax[i].axvline(x=40, color='r', linestyle='--')
    plt.show()


def test_intubation_effect():
    """Test that the intubation effect is visible on the signals."""
    assert George.dataframe['HR'].iloc[3 * 60] < George.dataframe['HR'].iloc[5 * 60]
    assert George.dataframe['SAP'].iloc[3 * 60] < George.dataframe['SAP'].iloc[5 * 60]
    assert George.dataframe['DAP'].iloc[3 * 60] < George.dataframe['DAP'].iloc[5 * 60]


def test_surgery_effect():
    """Test that the surgery effect is visible on the signals."""
    assert George.dataframe['HR'].iloc[40 * 60] < George.dataframe['HR'].iloc[42 * 60]
    assert George.dataframe['SAP'].iloc[40 * 60] < George.dataframe['SAP'].iloc[42 * 60]
    assert George.dataframe['DAP'].iloc[40 * 60] < George.dataframe['DAP'].iloc[42 * 60]
