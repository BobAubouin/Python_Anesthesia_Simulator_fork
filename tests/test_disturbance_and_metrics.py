import control
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from python_anesthesia_simulator import simulator, disturbances, metrics

ts = 60
age = 35
weight = 70
height = 170
gender = 0

# %%

George_1 = simulator.Patient([age, height, weight, gender], ts=ts,
                             model_propo="Schnider", model_remi="Minto", random_PD=False)
George_2 = simulator.Patient([age, height, weight, gender], ts=ts,
                             model_propo="Schnider", model_remi="Minto", random_PD=False)
George_3 = simulator.Patient([age, height, weight, gender], ts=ts,
                             model_propo="Schnider", model_remi="Minto", random_PD=False)

Ap = George_1.propo_pk.continuous_sys.A
Ar = George_1.remi_pk.continuous_sys.A

Bp = George_1.propo_pk.continuous_sys.B
Br = George_1.remi_pk.continuous_sys.B
A_nom = block_diag(Ap, Ar)
B_nom = block_diag(Bp, Br)
C = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
D = np.array([[0, 0]])

continuous_sys = control.ss(A_nom, B_nom, C, D)
discret_sys = continuous_sys.sample(ts)
A_nom = discret_sys.A
B_nom = discret_sys.B

# %% Simulation

N_simu = int(60 * 60/ts)


uP, uR = 0.13, 0.5
start_step = 20 * 60
end_step = 30 * 60
x = np.zeros((11, N_simu+1))
# George_1.save_data([0, 0, 0])
# George_2.save_data([0, 0, 0])
# George_3.save_data([0, 0, 0])
for index in range(N_simu):
    Dist_1 = disturbances.compute_disturbances(index*ts, dist_profil='realistic')
    George_1.one_step(uP, uR, dist=Dist_1, noise=False)
    x[:, index+1] = A_nom @ x[:, index] + B_nom @ np.array([uP, uR])
    Dist_2 = disturbances.compute_disturbances(index*ts, dist_profil='simple')
    George_2.one_step(uP, uR, dist=Dist_2, noise=False)
    Dist_3 = disturbances.compute_disturbances(index*ts, dist_profil='step', start_step=start_step, end_step=end_step)
    George_3.one_step(uP, uR, dist=Dist_3, noise=False)

# %% plots

if __name__ == '__main__':
    Time = George_1.dataframe['Time']/60
    fig, axs = plt.subplots(11, figsize=(14, 16))
    for i in range(5):
        axs[i].plot(George_1.dataframe['x_propo_' + str(i+1)], '-')
        axs[i].plot(x[i, :], '--')
        axs[i].set(xlabel='t', ylabel='$xp_' + str(i+1) + '$')
        plt.grid()
        axs[i+6].plot(George_1.dataframe['x_remi_' + str(i+1)], '-')
        axs[i+6].plot(x[i+6, :], '--')
        axs[i+6].set(xlabel='t', ylabel='$xr_' + str(i+1) + '$')
        plt.grid()

    axs[5].plot(George_1.dataframe['x_propo_' + str(5+1)], '-')
    axs[5].plot(x[5, :], '-')
    axs[5].set(xlabel='t', ylabel='$xp_' + str(5+1) + '$')
    plt.grid()

    plt.show()

    fig, ax = plt.subplots(3)
    ax[0].plot(Time, George_1.dataframe['u_propo'])
    ax[1].plot(Time, George_1.dataframe['u_remi'])
    ax[2].plot(Time, George_1.dataframe['u_nore'])

    ax[0].set_ylabel("Propo")
    ax[1].set_ylabel("Remi")
    ax[2].set_ylabel("Nore")

    plt.show()

    fig, ax = plt.subplots(4)

    ax[0].plot(Time, George_1.dataframe['BIS'])
    ax[1].plot(Time, George_1.dataframe['MAP'])
    ax[2].plot(Time, George_1.dataframe['CO'])
    ax[3].plot(Time, George_1.dataframe['TOL'])
    ax[0].plot(Time, George_2.dataframe['BIS'])
    ax[1].plot(Time, George_2.dataframe['MAP'])
    ax[2].plot(Time, George_2.dataframe['CO'])
    ax[3].plot(Time, George_2.dataframe['TOL'])
    ax[0].plot(Time, George_3.dataframe['BIS'])
    ax[1].plot(Time, George_3.dataframe['MAP'])
    ax[2].plot(Time, George_3.dataframe['CO'])
    ax[3].plot(Time, George_3.dataframe['TOL'])

    ax[0].set_ylabel("BIS")
    ax[1].set_ylabel("MAP")
    ax[2].set_ylabel("CO")
    ax[3].set_ylabel("TOL")
    ax[3].set_xlabel("Time (min)")
    for i in range(4):
        ax[i].grid()
    plt.show()

# %% metrics

metric_1 = metrics.compute_control_metrics(
    George_1.dataframe.loc[:10*60/ts, 'Time'],
    George_1.dataframe.loc[:10*60/ts, 'BIS'],
    phase='induction'
)
metric_2 = metrics.compute_control_metrics(
    George_2.dataframe.loc[:10*60/ts, 'Time'],
    George_2.dataframe.loc[:10*60/ts, 'BIS'],
    phase='induction'
)
metric_3 = metrics.compute_control_metrics(
    George_3.dataframe['Time'],
    George_3.dataframe['BIS'],
    phase='total',
    start_step=start_step,
    end_step=end_step
)

metric_1_new = metrics.new_metrics_induction(
    George_1.dataframe.loc[:10*60/ts, 'Time'].values,
    George_1.dataframe.loc[:10*60/ts, 'BIS'].values,
)

metric_3_new = metrics.new_metrics_maintenance(
    George_3.dataframe.loc[10*60/ts:, 'Time'].values,
    George_3.dataframe.loc[10*60/ts:, 'BIS'].values,
)


# %% test

def test_dynamic_simulation():
    assert np.allclose(George_1.dataframe['x_propo_1'], x[0, :])
    assert np.allclose(George_1.dataframe['x_propo_2'], x[1, :])
    assert np.allclose(George_1.dataframe['x_propo_3'], x[2, :])
    assert np.allclose(George_1.dataframe['x_propo_4'], x[3, :])
    assert np.allclose(George_1.dataframe['x_propo_5'], x[4, :])
    assert np.allclose(George_1.dataframe['x_propo_6'], x[5, :])
    assert np.allclose(George_1.dataframe['x_remi_1'], x[6, :])
    assert np.allclose(George_1.dataframe['x_remi_2'], x[7, :])
    assert np.allclose(George_1.dataframe['x_remi_3'], x[8, :])
    assert np.allclose(George_1.dataframe['x_remi_4'], x[9, :])
    assert np.allclose(George_1.dataframe['x_remi_5'], x[10, :])


def test_metrics():
    # No undershoot during induction
    assert metric_1['BIS_NADIR'].iloc[0] > 50
    assert metric_2['BIS_NADIR'].iloc[0] > 50

    assert np.allclose(metric_1['US'].iloc[0], 0.0)
    assert np.allclose(metric_2['US'].iloc[0], 0.0)

    # Time to target equal to 9 minutes
    assert np.allclose(metric_1['TT'].iloc[0], 9)
    assert np.allclose(metric_2['TT'].iloc[0], 9)
    assert np.allclose(metric_3['TT'].iloc[0], 9)

    # Settling time at 10% equal to 9 minutes
    assert np.allclose(metric_1['ST10'].iloc[0], 9)
    assert np.allclose(metric_2['ST10'].iloc[0], 9)
    assert np.allclose(metric_3['ST10'].iloc[0], 9)

    # Settling time at 20% equal to 6 minutes
    assert np.allclose(metric_1['ST20'].iloc[0], 6)
    assert np.allclose(metric_2['ST20'].iloc[0], 6)
    assert np.allclose(metric_3['ST20'].iloc[0], 6)

    # test maintenance phase
    assert np.allclose(metric_3['TTp'].iloc[0], 10)
    assert metric_3['BIS_NADIRp'].iloc[0] > 50
    assert np.isnan(metric_3['TTn'].iloc[0])
    assert metric_3['BIS_NADIRn'].iloc[0] < 50


def test_new_metrics():
    """test the new metrics."""

    assert (metric_1_new['Sleep_Time'] == 6).all()
    assert (metric_1_new["Low BIS time"] == 0).all()
    assert abs(metric_1_new['Lowest BIS'].iloc[0] - 53.2) <= 1e-1
    assert (metric_1_new['Settling time'] == 6).all()

    assert (metric_3_new['Time out of range'] == 0).all()
    assert abs(metric_3_new['Lowest BIS'].iloc[0] - 42.2) <= 1e-1
    assert abs(metric_3_new['Highest BIS'].iloc[0] - 57.0) <= 1e-1
