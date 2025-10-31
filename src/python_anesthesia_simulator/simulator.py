from typing import Optional
# Third party imports
import numpy as np
import pandas as pd
import casadi as cas
from scipy.signal import dlsim, TransferFunction
from .patient import Patient
from .tci_control import TCIController
from .disturbances import compute_disturbances


class Simulator:
    """Class to add environment and usefull functions for simulation."""

    def __init__(self,
                 patient: Patient = None,
                 tci_propo: Optional[str] = None,
                 tci_remi: Optional[str] = None,
                 #  tci_nore_: Optional[bool] = False,  not yet available
                 #  tci_atracurium: Optional[bool] = False, not yet available
                 disturbance_profil: Optional[str] = None,
                 noise: bool = False,
                 save_data: bool = True,
                 ):
        """Initialize the Simulator with a patient, and eventual TCI pumps.

        Parameters
        ----------
        patient: Patient
            The virtual patient object.
        tci_propo: str, optional 
            Type of TCI for Propofol. Can be either 'Plasma', 'Effect_site' or 'none'. Defaults to 'none'.
        tci_remi: str, optional
            Type of TCI for Remifentanil. Can be either 'Plasma', 'Effect_site' or 'none'. Defaults to 'none'.
        disturbance_profil: str, optional
            Type of disturbance profile to apply. See disturbance module for more details. The default is None.
        noise: bool, optional
            If True, add noise to the outputs of the patient model. The default is False.
        save_data: bool, optional
            If True, save the simulation data in a dataframe. The default is False.
        """
        self.patient = patient
        self.ts = patient.ts
        self.time = 0
        self.disturbance_profil = disturbance_profil
        self.noise = noise
        self.save_data = save_data
        self.demographic = [
            patient.age,
            patient.weight,
            patient.height,
            patient.sex
        ]
        if tci_propo != 'none':
            if tci_propo not in ['Plasma', 'Effect_site']:
                raise ValueError('tci_propo must be either "Plasma", "Effect_site" or "none"')
            self.tci_propo = TCIController(
                self.demographic,
                drug_name='Propofol',
                model_used=patient.model_propo,
                sampling_time=self.ts,
            )
        else:
            self.tci_propo = None
        if tci_remi != 'none':
            if tci_remi not in ['Plasma', 'Effect_site']:
                raise ValueError('tci_remi must be either "Plasma", "Effect_site" or "none"')
            self.tci_remi = TCIController(
                self.demographic,
                drug_name='Remifentanil',
                model_used=patient.model_remi,
                sampling_time=self.ts,
            )
        else:
            self.tci_remi = None

        # Initialize the buffer to simulate BIS delay
        self.bis_delay_buffer = np.ones(int(np.ceil(self.bis_delay_max / self.ts))) * self.bis

        # init noise model
        self.map_noise_std = 30
        self.hr_noise_std = 17
        self.bis_noise_std = 35
        # MAP
        xi = 2
        a = 4 * xi**2 - 2
        y = (-a + np.sqrt(a**2 + 4)) / 2
        omega = 0.01/np.sqrt(y)
        map_filter = TransferFunction([1], [1 / omega**2, 2 * xi / omega, 1])
        self.map_noise_filter = map_filter.to_discrete(self.ts, method='bilinear')
        # HR
        xi = 10
        a = 4 * xi**2 - 2
        y = (-a + np.sqrt(a**2 + 4)) / 2
        omega = 0.02/np.sqrt(y)
        hr_filter = TransferFunction([1], [1 / omega**2, 2 * xi / omega, 1])
        self.hr_noise_filter = hr_filter.to_discrete(self.ts, method='bilinear')
        # bis
        xi = 1
        a = 4 * xi**2 - 2
        y = (-a + np.sqrt(a**2 + 4)) / 2
        omega = 0.04/np.sqrt(y)
        bis_filter = TransferFunction([1], [1 / omega**2, 2 * xi / omega, 1])
        self.bis_noise_filter = bis_filter.to_discrete(self.ts, method='bilinear')

        white_noise_map = np.random.normal(0, self.map_noise_std, 1000)
        white_noise_hr = np.random.normal(0, self.hr_noise_std, 1000)
        white_noise_bis = np.random.normal(0, self.bis_noise_std, 1000)
        _, self.map_noise = dlsim(self.map_noise_filter, u=white_noise_map)
        _, self.hr_noise = dlsim(self.hr_noise_filter, u=white_noise_hr)
        _, self.bis_noise = dlsim(self.bis_noise_filter, u=white_noise_bis)
        self.noise_index = 0

        # Save data
        if self.save_data:
            self.init_dataframe()
            self.save_data()

    def one_step(self,
                 input_propo: float = 0,
                 input_remi: float = 0,
                 input_nore: float = 0,
                 input_atracurium: float = 0,
                 sqi: float = 100,
                 ) -> tuple[float, float, float, float]:
        r"""Simulate one step of the patient model with given inputs.

        If tci pumps are used, the inputs are the target concentrations. Otherwise, they are the infusion rates.

        Parameters
        ----------
        input_propo : float, optional
            Infusion rate (mg/s) or target concentration (µg/ml) for Propofol. The default is 0.
        input_remi : float, optional
            Infusion rate (µg/s) or target concentration (ng/ml) for Remifentanil. The default is 0.
        input_nore : float, optional
            Infusion rate (µg/s) for Norepinephrine. The default is 0.
        input_atracurium : float, optional
            Infusion rate (mg/s) for Atracurium. The default is 0.
        sqi: float, optional
            Signal Quality Index of the BIS signal. It affects the BIS delay (expressed in seconds) according to the relationship proposed in [Wahlquist2025]_: :math:`bis\_delay = bis\_delay\_max * (1 - \frac{sqi}{100})`. The default is 100.

        Returns
        -------
        tuple[float, float, float, float]
            BIS, MAP, HR, NMB values after one step of simulation.
        """
        if self.tci_propo is not None:
            infusion_propo = self.tci_propo.one_step(target=input_propo)
        else:
            infusion_propo = input_propo
        if self.tci_remi is not None:
            infusion_remi = self.tci_remi.one_step(target=input_remi)
        else:
            infusion_remi = input_remi

        disturbances = compute_disturbances(
            time=self.time,
            dist_profil=self.disturbance_profil,
        )
        self.patient.one_step(
            u_propo=infusion_propo,
            u_remi=infusion_remi,
            u_nore=input_nore,
            u_atra=input_atracurium,
            disturbances=disturbances,
        )

        # add noise
        if self.noise:
            self.add_noise()

        # bis delay
        delay = self.bis_delay_max * (1 - sqi / 100)
        delay_steps = int(np.ceil(delay / self.ts)) - 1
        if delay_steps > 0:

            # Approximated by excess
            self.bis_delay_buffer = np.roll(self.bis_delay_buffer, -1)
            self.bis_delay_buffer[delay_steps:] = [self.bis] * len(self.bis_delay_buffer[delay_steps:])
            self.bis = self.bis_delay_buffer[0]
        else:
            self.bis_delay_buffer = np.ones(int(np.ceil(self.bis_delay_max / self.ts))) * self.bis

        self.time += self.ts

        if self.save_data:
            self.save_data(
                inputs=[
                    infusion_propo,
                    infusion_remi,
                    input_nore,
                    input_atracurium,
                    sqi,
                ]
            )
        return self.patient.bis, self.patient.map, self.patient.hr, self.patient.nmb

    def add_noise(self):
        r"""
        Add noise on MAP, HR and BIS.

        All noise are considered white noise filtered by a second order transfert function:

        - For MAP, the standard deviation of the white noise is 30 and the filter is a second-order low-pass noise filter with damping ratio ξ=2 and cutoff frequency ω=0.01.
        - For HR, the standard deviation of the white noise is 17 and the filter is a second-order low-pass noise filter with damping ratio ξ=10 and cutoff frequency ω=0.02. In addition, the output is ceiled to the nearest integer.
        - For BIS, the standard deviation of the white noise is 35 and the filter is a second-order low-pass noise filter with damping ratio ξ=1 and cutoff frequency ω=0.04.

        See identification details on this `Notebook <https://github.com/AnesthesiaSimulation/PAS_vs_vitalDB/blob/main/scripts/identify_noise.ipynb>`_

        """
        self.noise_index += 1
        if self.noise_index >= len(self.bis_noise):
            # new list noise
            white_noise_map = np.random.normal(0, self.map_noise_std, 1000)
            white_noise_hr = np.random.normal(0, self.hr_noise_std, 1000)
            white_noise_bis = np.random.normal(0, self.bis_noise_std, 1000)
            _, self.map_noise = dlsim(self.map_noise_filter, u=white_noise_map)
            _, self.hr_noise = dlsim(self.hr_noise_filter, u=white_noise_hr)
            _, self.bis_noise = dlsim(self.bis_noise_filter, u=white_noise_bis)
            self.noise_index = 0

        self.bis += self.bis_noise[self.noise_index]
        self.bis = np.clip(self.bis, 0, 100)
        self.map += self.map_noise[self.noise_index]
        self.hr += self.hr_noise[self.noise_index]
        self.hr = np.ceil(self.hr)

    def init_dataframe(self):
        r"""Initilize the dataframe variable with the following columns:

            - 'Time': Simulation time (s)
            - 'BIS': Bispectral Index
            - 'SQI': Signal Quality Index
            - 'TOL': Tolerance level
            - 'NMB': Neuromuscular blockade level (%)
            - 'TPR': Total eripheral resistance (mmHg min/ mL) 
            - 'SV': Stroke volume (ml)
            - 'HR': Heart rate (beat/min)
            - 'MAP': Mean Arterial Pressure (mmHg)
            - 'CO': Cardiac Output (L/min)
            - 'u_propo': Propofol infusion rate (mg/s)
            - 'u_remi': Remifentanil infusion rate (µg/s)
            - 'u_nore': Norepinephrine infusion rate (µg/s)
            - 'u_atra': Atracurium infusion rate (µg/s)
            - 'x_propo_1' to 'x_propo_4': States of the propofol PK model
            - 'x_remi_1' to 'x_remi_4': States of the remifentanil PK model
            - 'x_nore': State of the norepinephrine PK model
            - 'x_atra_1' to 'x_atra_4': States of the atracurium PK model
            - 'blood_volume': Blood volume (L)
            if applicable TCI targets are also added:
            - 'target_propo': Target concentration for Propofol (µg/ml)
            - 'target_remi': Target concentration for Remifentanil (ng/ml)

        """
        self.Time = 0
        column_names = ['Time',  # time
                        'BIS', 'SQI', 'TOL', 'NMB', 'MAP', 'CO',  # outputs
                        'TPR', 'SV', 'HR', 'SAP', 'DAP',  # outputs
                        'u_propo', 'u_remi', 'u_nore', 'u_atra',  # inputs
                        'blood_volume']  # nore concentration and blood volume
        propo_state_names = [f'x_propo_{i + 1}' for i in range(len(self.propo_pk.x))]
        remi_state_names = [f'x_remi_{i + 1}' for i in range(len(self.remi_pk.x))]
        nore_state_names = [f'x_nore_{i + 1}' for i in range(len(self.nore_pk.x))]
        atra_state_names = [f'x_atra_{i + 1}' for i in range(len(self.atracurium_pk.x))]
        column_names += propo_state_names + remi_state_names + nore_state_names + atra_state_names
        if self.tci_propo is not None:
            column_names.append('target_propo')
        if self.tci_remi is not None:
            column_names.append('target_remi')
        self.dataframe = pd.DataFrame(columns=column_names, dtype=float)

    def save_data(self, inputs: list = [0, 0, 0, 0, 100]):
        r"""Save all current internal variables as a new line in self.dataframe."""
        # store data
        new_line = {'Time': self.time,
                    'BIS': self.patient.bis,  # measures
                    'TOL': self.patient.tol,
                    'TPR': self.patient.tpr,
                    'NMB': self.patient.nmb,
                    'SV': self.patient.sv,
                    'HR': self.patient.hr,
                    'MAP': self.patient.map,
                    'CO': self.patient.co,
                    'SAP': self.patient.sap,
                    'DAP': self.patient.dap,
                    'u_propo': inputs[0],  # inputs
                    'u_remi': inputs[1],
                    'u_nore': inputs[2],
                    'u_atra': inputs[3],
                    'SQI': inputs[4],
                    'blood_volume': self.blood_volume}  # blood volume

        line_x_propo = {f'x_propo_{i + 1}': self.propo_pk.x[i, 0] for i in range(len(self.propo_pk.x))}
        line_x_remi = {f'x_remi_{i + 1}': self.remi_pk.x[i, 0] for i in range(len(self.remi_pk.x))}
        line_x_nore = {f'x_nore_{i + 1}': self.nore_pk.x[i, 0] for i in range(len(self.nore_pk.x))}
        line_x_atra = {f'x_atra_{i + 1}': self.atracurium_pk.x[i, 0] for i in range(len(self.atracurium_pk.x))}
        new_line.update(line_x_propo)
        new_line.update(line_x_remi)
        new_line.update(line_x_nore)
        new_line.update(line_x_atra)
        if self.tci_propo is not None:
            new_line['target_propo'] = self.tci_propo.target
        if self.tci_remi is not None:
            new_line['target_remi'] = self.tci_remi.target
        self.dataframe = pd.concat(
            [df for df in (self.dataframe, pd.DataFrame(new_line, index=[1], dtype=float)) if not df.empty],
            ignore_index=True
        )

    def generate_random_patient(self,
                                distribution: str = 'uniform',
                                patient_arg: list = [],
                                ):
        """
        Generate a random patient with caracteristique following either a uniform distribution or a distribution fitted on VitalDB data.

        Paramters
        ----------
        distribution: str, optionnal
            Choose how patient characteristic are drawn. Can be "uniform" or "VitalDB". Default is "uniform".
        patient_arg: list, optionnal
            Arguments to pass to init the patient class. Default is an empty list.
        Return
        -------
        patient: Patient object
            Instance of the patient class.
        """
        if distribution == 'uniform':
            age = np.random.randint(low=18, high=81)
            height = np.random.randint(low=150, high=190)
            weight = np.random.randint(low=50, high=100)
            sex = np.random.randint(low=0, high=2)
        elif distribution == 'VitalDB':

            mean_male = [55.4, 167.6, 65.8]  # age, height, weight
            mean_female = [51.6, 157.7, 54.4]
            sigma_male = [
                [116.2, -13.9, -41.4],
                [-13.9,  33.3,  18.1],
                [-41.4,  18.1, 105.0]]
            sigma_female = [
                [251.5, -44.0, -19.0],
                [-44.0,  21.3,  22.5],
                [-19.0,  22.5,  68.8]]
            sex = np.random.randint(low=0, high=2)
            good_range = False
            while not good_range:
                if sex == 0:
                    vec = np.random.multivariate_normal(mean_female, sigma_female)
                else:
                    vec = np.random.multivariate_normal(mean_male, sigma_male)
                good_age = (vec[0] >= 18) and (vec[0] <= 80)
                good_height = (vec[1] >= 145) and (vec[1] <= 185)
                good_weight = (vec[2] >= 40) and (vec[2] <= 95)
                good_range = good_age and good_height and good_weight
            age = vec[0]
            height = vec[1]
            weight = vec[2]

        self.patient = Patient(
            [age, height, weight, sex],
            *patient_arg,
        )
        return self.patient
