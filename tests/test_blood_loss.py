import matplotlib.pyplot as plt
from python_anesthesia_simulator import Patient, TCIController

# %% Initialization patient
ts = 5
age, height, weight, sex = 74, 164, 88, 1
George = Patient([age, height, weight, sex], ts=ts,
                 model_propo="Eleveld", model_remi="Eleveld", co_update=True)

tci_propo = TCIController(
    [age, height, weight, sex],
    drug_name="Propofol",
    drug_concentration=10,
    sampling_time=ts,
    model_used="Eleveld",
)
tci_remi = TCIController(
    [age, height, weight, sex],
    drug_name="Remifentanil",
    drug_concentration=10,
    sampling_time=ts,
    model_used="Eleveld",
)


# %% Simulation

N_simu = int(120 * 60/ts)

uN = 0
target_propo = 3.0
target_remi = 2
blood_loss_rate = 200  # ml/min
blood_gain_rate = 50  # ml/min
time_start_bleeding = 61 * 60  # seconds
time_end_bleeding = 71 * 60  # seconds
time_start_transfusion = 75 * 60  #
time_end_transfusion = 115 * 60  # seconds
for index in range(N_simu):
    uP = tci_propo.one_step(target_propo)/3600 * 10  # convert to mg/s
    uR = tci_remi.one_step(target_remi)/3600 * 10  # convert to mg/s

    if index*ts > time_start_bleeding and index*ts < time_end_bleeding:
        Bis, Co, Map, Tol, Nmb = George.one_step(u_propo=uP, u_remi=uR, u_nore=uN,
                                                 blood_rate=- blood_loss_rate, noise=False)
    elif index*ts > time_start_transfusion and index*ts < time_end_transfusion:
        Bis, Co, Map, Tol, Nmb = George.one_step(u_propo=uP, u_remi=uR, u_nore=uN,
                                                 blood_rate=blood_gain_rate, noise=False)
    else:
        Bis, Co, Map, Tol, Nmb = George.one_step(u_propo=uP, u_remi=uR, u_nore=uN,
                                                 blood_rate=0, noise=False)


# %% test
index_start_bleeding = int(time_start_bleeding/ts)
index_end_bleeding = int(time_end_bleeding/ts)
index_start_transfusion = int(time_start_transfusion/ts)
index_end_transfusion = int(time_end_transfusion/ts)


def test_bleeding_effect():
    """if bleeding is not stopped, BIS, MAP and CO should decrease, TOL and drugs concentration should increase."""
    assert George.dataframe['x_propo_1'][index_start_bleeding] < George.dataframe['x_propo_1'][index_end_bleeding]
    assert George.dataframe['x_remi_1'][index_start_bleeding] < George.dataframe['x_remi_1'][index_end_bleeding]
    assert George.dataframe['BIS'][0] > George.dataframe['BIS'][index_end_bleeding]
    assert George.dataframe['MAP'][0] > George.dataframe['MAP'][index_end_bleeding]
    assert George.dataframe['CO'][0] > George.dataframe['CO'][index_end_bleeding]
    assert George.dataframe['TOL'][0] < George.dataframe['TOL'][index_end_bleeding]


def test_stop_bleeding_effect():
    """ if transfusion is not stopped, BIS, MAP and CO should increase, and TOL should decrease."""
    assert George.dataframe['BIS'][index_start_transfusion] < George.dataframe['BIS'][index_end_transfusion]
    assert George.dataframe['MAP'][index_start_transfusion] < George.dataframe['MAP'][index_end_transfusion]
    assert George.dataframe['CO'][index_start_transfusion] < George.dataframe['CO'][index_end_transfusion]
    assert George.dataframe['TOL'][index_start_transfusion] > George.dataframe['TOL'][index_end_transfusion]


# %% plot
if __name__ == '__main__':
    fig, ax = plt.subplots(3)
    Time = George.dataframe['Time']/60
    ax[0].plot(Time, George.dataframe['u_propo'])
    ax[1].plot(Time, George.dataframe['u_remi'])
    ax[2].plot(Time, George.dataframe['u_nore'])

    ax[0].set_ylabel("Propo")
    ax[1].set_ylabel("Remi")
    ax[2].set_ylabel("Nore")
    for i in range(3):
        ax[i].grid()
    plt.ticklabel_format(style='plain')
    plt.show()

    fig, ax = plt.subplots(1)

    ax.plot(Time, George.dataframe['x_propo_4'], label="Propofol")
    ax.plot(Time, George.dataframe['x_remi_4'], label="Remifentanil")
    ax.plot(Time, George.dataframe['x_nore_1'], label="Norepinephrine")
    plt.title("Hypnotic effect site Concentration")
    ax.set_xlabel("Time (min)")
    plt.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(5, figsize=(10, 10))

    ax[0].plot(Time, George.dataframe['BIS'], label="BIS")
    ax[0].plot(Time, George.dataframe['TOL']*100, label="TOL (x100)")
    ax[0].legend()
    ax[1].plot(Time, George.dataframe['MAP'])
    ax[2].plot(Time, George.dataframe['CO'])
    ax[3].plot(Time, George.dataframe['HR'], label="HR")
    ax[3].plot(Time, George.dataframe['SV'], label="SV")
    ax[3].legend()
    ax[4].plot(Time, George.dataframe['blood_volume'])

    ax[1].set_ylabel("MAP")
    ax[2].set_ylabel("CO")
    ax[4].set_ylabel("blood volume")
    ax[4].set_xlabel("Time (min)")
    for i in range(5):
        ax[i].grid()
    plt.ticklabel_format(style='plain')
    plt.show()

    # test
    test_bleeding_effect()
    test_stop_bleeding_effect()
    print("All tests passed successfully.")
