import matplotlib.pyplot as plt
from python_anesthesia_simulator import Patient, TCIController


age = 28
height = 165
weight = 65
sex = 0
patient_info = [age, height, weight, sex]

sampling_time = 2
propofol_target = 4
remifentanil_target = 3


# init the patient simulation
patient = Patient(
    patient_info,
    model_propo='Schnider',
    model_remi='Minto'
)

# initialize TCI
concentration = 10  # mg/ml for Propofol and µg/ml for remifentanil
tci_propo = TCIController(
    patient_info,
    drug_name='Propofol',
    model_used="Schnider",
    drug_concentration=concentration
)
tci_remi = TCIController(
    patient_info,
    drug_name='Remifentanil',
    model_used="Minto",
    drug_concentration=concentration
)

N_simu = 5 * 60 // sampling_time  # 10 minutes


for time_step in range(N_simu):
    u_propo = tci_propo.one_step(propofol_target)  # in ml/h
    u_remi = tci_remi.one_step(remifentanil_target)  # in ml/h

    patient.one_step(u_propo / 3600 * concentration, u_remi / 3600 * concentration)

if __name__ == "__main__":
    plt.subplot(2, 1, 1)
    plt.plot(patient.dataframe['Time'] / 60, patient.dataframe['u_propo'], label='Propofol (mg/s)')
    plt.plot(patient.dataframe['Time'] / 60, patient.dataframe['u_remi'], label='Remifentanil (ug/s)')
    plt.ylabel('Drug rate')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(patient.dataframe['Time'] / 60, patient.dataframe['x_propo_4'], label='Propofol (ug/ml)')
    plt.plot(patient.dataframe['Time'] / 60, patient.dataframe['x_remi_4'], label='Remifentanil (ng/ml)')
    # plot target
    plt.plot(patient.dataframe['Time'] / 60, [propofol_target] *
             len(patient.dataframe['Time']), '--', label='Propofol target')
    plt.plot(patient.dataframe['Time'] / 60, [remifentanil_target] *
             len(patient.dataframe['Time']), '--', label='Remifentanil target')
    plt.ylabel('Effect site concentration ')
    plt.xlabel('Time (min)')
    plt.legend()
    plt.grid()

    plt.show()


# test

def test_tci_ouput_range():
    """ensure that the command belong in the acceptable range."""
    assert (patient.dataframe['u_propo'] >= 0).all()
    assert (patient.dataframe['u_propo'] <= tci_propo.infusion_max).all()
    assert (patient.dataframe['u_remi'] >= 0).all()
    assert (patient.dataframe['u_remi'] <= tci_remi.infusion_max).all()


def test_tci_behavior():
    # ensure that the concentration reach the target (maximum of 1%)
    assert patient.dataframe['x_propo_4'].iloc[-1] <= propofol_target * 1.01
    assert patient.dataframe['x_propo_4'].iloc[-1] >= propofol_target * 0.99
    assert patient.dataframe['x_remi_4'].iloc[-1] <= remifentanil_target * 1.01
    assert patient.dataframe['x_remi_4'].iloc[-1] >= remifentanil_target * 0.99

    # ensure that there is not too much overshoot (maximum 5%)
    assert (patient.dataframe['x_propo_4'].iloc[-1] <= propofol_target * 1.05).all()
    assert (patient.dataframe['x_remi_4'].iloc[-1] <= remifentanil_target * 1.05).all()
