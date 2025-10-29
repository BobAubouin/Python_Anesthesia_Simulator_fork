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
                 patient: Patient,
                 tci_propo: Optional[str] = None,
                 tci_remi: Optional[str] = None,
                 #  tci_nore_: Optional[bool] = False,  not yet available
                 #  tci_atracurium: Optional[bool] = False, not yet available
                 disturbance_profil: Optional[str] = None,
                 noise: bool = False,
                 save_data: bool = False,
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
        self.demographic = [
            patient.age,
            patient.weight,
            patient.height,
            patient.gender
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
        self.bis_noise_std = 3
        self.co_noise_std = 0.1
        self.map_noise_std = 5
        xi = 0.2
        target_peak_fr = 0.03 * 2 * np.pi
        omega = target_peak_fr / np.sqrt(1 - 2 * xi**2)
        noise_filter = TransferFunction([0.1, 1], [1 / omega**2, 2 * xi / omega, 1])
        self.noise_filter_d = noise_filter.to_discrete(self.ts, method='bilinear')
        white_noise = np.random.normal(0, self.bis_noise_std, 1000)
        _, self.bis_noise = dlsim(self.noise_filter_d, u=white_noise)
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

        if tci pumps are used, the inputs are the target concentrations. Otherwise, they are the infusion rates.
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
        Add noise on the outputs of the model (except TOL and NMB).

        The MAP and CO noises are considered white noise while the BIS noise is filtered.
        The filter of the BIS noise is a second order low pass filter with a cut-off frequency of 0.03 Hz.

        """
        # compute filter noise for BIS
        # white noise
        self.noise_index += 1
        if self.noise_index >= len(self.bis_noise):
            self.noise_index = 0
            # new list noise
            white_noise = np.random.normal(0, self.bis_noise_std, 1000)
            _, self.bis_noise = dlsim(self.noise_filter_d, u=white_noise)
        self.bis += self.bis_noise[self.noise_index]
        self.bis = np.clip(self.bis, 0, 100)
        # random noise for MAP and CO
        self.map += np.random.normal(scale=self.map_noise_std)
        self.co += np.random.normal(scale=self.co_noise_std)

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
