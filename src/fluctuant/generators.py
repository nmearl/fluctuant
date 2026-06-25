import numpy as np
import matplotlib.pyplot as plt


class SyntheticTransientGenerator:
    """Synthetic AGN and TDE light curve generator."""

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_agn_with_components(self, cadence_days=3, duration_days=1000,
                                     baseline_mag=19.0, noise_level=0.05,
                                     include_flare=True):
        """
        Generate a single AGN light curve with all model components.

        Returns
        -------
        dict
            Keys: 'times', 'mags', 'errors', 'drw_model', 'flare_model',
            'full_model', 'baseline_mag'.
        """
        # Generate irregular observation times
        times = self.generate_irregular_times(duration_days, cadence_days)
        n_obs = len(times)

        # DRW parameters
        tau = np.random.uniform(50, 300)
        sf_inf = np.random.uniform(0.05, 0.15)

        # Generate DRW baseline
        drw_mags = self.damped_random_walk(times, tau, sf_inf, baseline_mag)

        # Add flare if requested
        if include_flare:
            # Get flare model (deterministic component)
            flare_model = self.get_flare_model(times)
            full_model = drw_mags - flare_model  # Subtract for magnitude
        else:
            flare_model = np.zeros_like(times)
            full_model = drw_mags.copy()

        # Add photometric noise to get observed magnitudes
        errors = np.random.uniform(0.03, noise_level, n_obs)
        observed_mags = full_model + np.random.normal(0, errors)

        return {
            'times': times,
            'mags': observed_mags,
            'errors': errors,
            'drw_model': drw_mags,
            'flare_model': flare_model,
            'full_model': full_model,
            'baseline_mag': baseline_mag
        }

    def generate_tde_with_components(self, cadence_days=3, duration_days=1000,
                                     baseline_mag=19.0, noise_level=0.05):
        """
        Generate a single TDE light curve with all model components.

        Returns
        -------
        dict
            Keys: 'times', 'mags', 'errors', 'model', 'baseline_mag'.
        """
        times = self.generate_irregular_times(duration_days, cadence_days)
        n_obs = len(times)

        # TDE parameters
        t0 = np.random.uniform(100, 400)
        rise_time = np.random.uniform(10, 40)
        amplitude = np.random.uniform(1.5, 4.0)
        decay_power = np.random.uniform(-5 / 3, -1.0)

        # Generate TDE model
        model_mags = self.tde_lightcurve(times, t0, rise_time, amplitude,
                                         decay_power, baseline_mag)

        # Add photometric noise
        errors = np.random.uniform(0.03, noise_level, n_obs)
        observed_mags = model_mags + np.random.normal(0, errors)

        return {
            'times': times,
            'mags': observed_mags,
            'errors': errors,
            'model': model_mags,
            'baseline_mag': baseline_mag
        }

    def get_flare_model(self, times):
        """Flare magnitude delta (no noise), amplitude > 0 means brightening."""
        # Random flare time
        flare_start = np.random.uniform(times.min() + 100, times.max() - 200)

        # Flare parameters
        amplitude = np.random.uniform(0.5, 1.5)
        rise_time = np.random.uniform(5, 20)
        decay_time = np.random.uniform(20, 80)

        # Create flare profile
        flare = np.zeros_like(times)
        peak_time = flare_start + rise_time

        for i, t in enumerate(times):
            if t < flare_start:
                flare[i] = 0
            elif t < peak_time:
                dt = t - flare_start
                progress = dt / rise_time
                flare[i] = amplitude * progress ** 2
            else:
                dt = t - peak_time
                flare[i] = amplitude * np.exp(-dt / decay_time)

        return flare

    def generate_agn(self, n_objects=100, cadence_days=3, duration_days=1000,
                     baseline_mag=19.0, noise_level=0.05,
                     include_flares=True, flare_fraction=0.3):
        """Generate multiple AGN light curves (returns simple format for Astromer)."""
        light_curves = []

        for i in range(n_objects):
            times = self.generate_irregular_times(duration_days, cadence_days)
            n_obs = len(times)

            tau = np.random.uniform(50, 300)
            sf_inf = np.random.uniform(0.05, 0.15)

            mags = self.damped_random_walk(times, tau, sf_inf, baseline_mag)

            if include_flares and np.random.random() < flare_fraction:
                mags = self.add_agn_flare(times, mags)

            errors = np.random.uniform(0.03, noise_level, n_obs)
            mags += np.random.normal(0, errors)

            lc = np.column_stack([times, mags, errors])
            light_curves.append(lc)

        return light_curves

    def add_agn_flare(self, times, mags):
        """Add flare to existing magnitude array."""
        flare_model = self.get_flare_model(times)
        return mags - flare_model

    def damped_random_walk(self, times, tau, sf_inf, baseline_mag):
        """Generate DRW process."""
        n = len(times)
        mags = np.zeros(n)
        mags[0] = baseline_mag

        for i in range(1, n):
            dt = times[i] - times[i - 1]
            decay = np.exp(-dt / tau)
            mean = (mags[i - 1] - baseline_mag) * decay
            variance = sf_inf ** 2 * (1 - decay ** 2)
            mags[i] = baseline_mag + mean + np.random.normal(0, np.sqrt(variance))

        return mags

    def generate_irregular_times(self, duration, mean_cadence):
        """Generate realistic irregular observation times."""
        times = []
        current_time = 0

        while current_time < duration:
            if np.random.random() < 0.1:
                gap = np.random.uniform(20, 60)
            else:
                gap = np.random.exponential(mean_cadence)

            current_time += gap
            if current_time < duration:
                times.append(current_time)

        return np.array(times)

    def generate_tde(self, n_objects=100, cadence_days=3, duration_days=1000,
                     baseline_mag=19.0, noise_level=0.05):
        """Generate multiple TDE light curves."""
        light_curves = []

        for i in range(n_objects):
            times = self.generate_irregular_times(duration_days, cadence_days)
            n_obs = len(times)

            t0 = np.random.uniform(100, 400)
            rise_time = np.random.uniform(10, 40)
            amplitude = np.random.uniform(1.5, 4.0)
            decay_power = np.random.uniform(-5 / 3, -1.0)

            mags = self.tde_lightcurve(times, t0, rise_time, amplitude,
                                       decay_power, baseline_mag)

            errors = np.random.uniform(0.03, noise_level, n_obs)
            mags += np.random.normal(0, errors)

            lc = np.column_stack([times, mags, errors])
            light_curves.append(lc)

        return light_curves

    def tde_lightcurve(self, times, t0, rise_time, amplitude, decay_power, baseline_mag):
        """Generate TDE light curve."""
        mags = np.full_like(times, baseline_mag)

        for i, t in enumerate(times):
            dt = t - t0

            if dt < 0:
                mags[i] = baseline_mag
            elif dt < rise_time:
                phase = dt / rise_time
                mags[i] = baseline_mag - amplitude * (1 - np.cos(np.pi * phase)) / 2
            else:
                t_since_peak = dt - rise_time
                power_law = amplitude * (1 + t_since_peak / 50) ** decay_power
                exp_decay = amplitude * np.exp(-t_since_peak / 200)
                weight = 1 / (1 + (t_since_peak / 100) ** 2)
                mags[i] = baseline_mag - (weight * power_law + (1 - weight) * exp_decay)

        return mags

    def generate_agn_multiband(self, n_objects=100, cadence_days=3, duration_days=1000,
                               baseline_mag=19.0, noise_level=0.05,
                               include_flares=True, flare_fraction=0.3):
        """
        Generate AGN light curves in g and r bands.

        r-band variability is 75% of g-band amplitude; r is ~0.3 mag brighter.
        r observations are offset by 1 day (ZTF alternating cadence).

        Parameters
        ----------
        n_objects : int
        cadence_days : float
        duration_days : float
        baseline_mag : float
            g-band quiescent magnitude.
        noise_level : float
        include_flares : bool
        flare_fraction : float

        Returns
        -------
        list of dict
            Each dict has keys ``'g'`` and ``'r'``, each an ndarray of
            shape ``(n_obs, 3)`` with columns [time, mag, err].
        """
        light_curves = []
        for _ in range(n_objects):
            times_g = self.generate_irregular_times(duration_days, cadence_days)
            times_r = times_g + 1.0
            n_obs = len(times_g)

            tau = np.random.uniform(50, 300)
            sf_inf_g = np.random.uniform(0.05, 0.15)
            sf_inf_r = sf_inf_g * 0.75

            mags_g = self.damped_random_walk(times_g, tau, sf_inf_g, baseline_mag)
            mags_r = self.damped_random_walk(times_r, tau, sf_inf_r, baseline_mag - 0.3)

            if include_flares and np.random.random() < flare_fraction:
                flare = self.get_flare_model(times_g)
                mags_g = mags_g - flare
                mags_r = mags_r - flare * 0.8

            errors_g = np.random.uniform(0.03, noise_level, n_obs)
            errors_r = np.random.uniform(0.03, noise_level, n_obs)
            mags_g += np.random.normal(0, errors_g)
            mags_r += np.random.normal(0, errors_r)

            light_curves.append({
                'g': np.column_stack([times_g, mags_g, errors_g]),
                'r': np.column_stack([times_r, mags_r, errors_r]),
            })
        return light_curves

    def generate_tde_multiband(self, n_objects=100, cadence_days=3, duration_days=1000,
                               baseline_mag=19.0, noise_level=0.05):
        """
        Generate TDE light curves in g and r bands.

        TDEs are bluer than AGN: r-band flare amplitude is ~55% of g-band.
        r observations are offset by 1 day (ZTF alternating cadence).

        Parameters
        ----------
        n_objects : int
        cadence_days : float
        duration_days : float
        baseline_mag : float
            g-band quiescent magnitude.
        noise_level : float

        Returns
        -------
        list of dict
            Each dict has keys ``'g'`` and ``'r'``, each an ndarray of
            shape ``(n_obs, 3)`` with columns [time, mag, err].
        """
        light_curves = []
        for _ in range(n_objects):
            times_g = self.generate_irregular_times(duration_days, cadence_days)
            times_r = times_g + 1.0
            n_obs = len(times_g)

            t0 = np.random.uniform(100, 400)
            rise_time = np.random.uniform(10, 40)
            amplitude_g = np.random.uniform(1.5, 4.0)
            amplitude_r = amplitude_g * np.random.uniform(0.5, 0.6)
            decay_power = np.random.uniform(-5 / 3, -1.0)

            mags_g = self.tde_lightcurve(times_g, t0, rise_time, amplitude_g,
                                          decay_power, baseline_mag)
            mags_r = self.tde_lightcurve(times_r, t0, rise_time, amplitude_r,
                                          decay_power, baseline_mag - 0.2)

            errors_g = np.random.uniform(0.03, noise_level, n_obs)
            errors_r = np.random.uniform(0.03, noise_level, n_obs)
            mags_g += np.random.normal(0, errors_g)
            mags_r += np.random.normal(0, errors_r)

            light_curves.append({
                'g': np.column_stack([times_g, mags_g, errors_g]),
                'r': np.column_stack([times_r, mags_r, errors_r]),
            })
        return light_curves

    def plot_with_model_overlay(self, n_examples=3):
        """Plot AGN and TDE light curves with model components overlaid."""
        fig, axes = plt.subplots(2, n_examples, figsize=(15, 10))

        for i in range(n_examples):
            # Generate AGN with components
            agn = self.generate_agn_with_components(include_flare=True)

            # Plot AGN
            ax = axes[0, i]

            # Data points with error bars
            ax.errorbar(agn['times'], agn['mags'], yerr=agn['errors'],
                        fmt='o', markersize=4, alpha=0.4, color='black',
                        label='Observed', zorder=1)

            # DRW baseline model (smooth line)
            ax.plot(agn['times'], agn['drw_model'], '-',
                    linewidth=2, alpha=0.7, color='blue',
                    label='DRW baseline', zorder=2)

            # Flare component (show as change from baseline)
            # Plot where baseline would be without flare
            ax.axhline(agn['baseline_mag'], color='gray',
                       linestyle='--', alpha=0.3, linewidth=1)

            # Full model (DRW + flare)
            ax.plot(agn['times'], agn['full_model'], '-',
                    linewidth=2.5, alpha=0.8, color='red',
                    label='DRW + Flare model', zorder=3)

            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'AGN {i + 1}')
            ax.invert_yaxis()
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)

            # Generate TDE with components
            tde = self.generate_tde_with_components()

            # Plot TDE
            ax = axes[1, i]

            # Data points with error bars
            ax.errorbar(tde['times'], tde['mags'], yerr=tde['errors'],
                        fmt='o', markersize=4, alpha=0.4, color='black',
                        label='Observed', zorder=1)

            # Baseline
            ax.axhline(tde['baseline_mag'], color='gray',
                       linestyle='--', alpha=0.3, linewidth=1,
                       label='Quiescent')

            # TDE model
            ax.plot(tde['times'], tde['model'], '-',
                    linewidth=2.5, alpha=0.8, color='orange',
                    label='TDE model', zorder=2)

            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'TDE {i + 1}')
            ax.invert_yaxis()
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_agn_detailed(self, n_examples=3):
        """Four-panel AGN plot: DRW, flare component, combined model, observed."""
        fig, axes = plt.subplots(n_examples, 4, figsize=(20, 4 * n_examples))

        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_examples):
            agn = self.generate_agn_with_components(include_flare=True)

            # Panel 1: DRW baseline only
            axes[i, 0].plot(agn['times'], agn['drw_model'],
                            'o-', markersize=3, alpha=0.6, color='blue')
            axes[i, 0].axhline(agn['baseline_mag'], color='gray',
                               linestyle='--', alpha=0.5)
            axes[i, 0].set_ylabel('Magnitude')
            axes[i, 0].set_title('DRW Baseline')
            axes[i, 0].invert_yaxis()
            axes[i, 0].grid(alpha=0.3)

            # Panel 2: Flare component (as magnitude change)
            axes[i, 1].plot(agn['times'], -agn['flare_model'],  # Negative because mag
                            'o-', markersize=3, alpha=0.6, color='orange')
            axes[i, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
            axes[i, 1].set_ylabel('Δ Magnitude')
            axes[i, 1].set_title('Flare Component')
            axes[i, 1].grid(alpha=0.3)

            # Panel 3: Combined model (DRW + Flare)
            axes[i, 2].plot(agn['times'], agn['drw_model'],
                            '-', linewidth=1.5, alpha=0.5,
                            color='blue', label='DRW')
            axes[i, 2].plot(agn['times'], agn['full_model'],
                            '-', linewidth=2, alpha=0.8,
                            color='red', label='DRW + Flare')
            axes[i, 2].set_ylabel('Magnitude')
            axes[i, 2].set_title('Combined Model')
            axes[i, 2].invert_yaxis()
            axes[i, 2].legend(fontsize=8)
            axes[i, 2].grid(alpha=0.3)

            # Panel 4: Observed data + model
            axes[i, 3].errorbar(agn['times'], agn['mags'],
                                yerr=agn['errors'],
                                fmt='o', markersize=3, alpha=0.4,
                                color='black', label='Observed')
            axes[i, 3].plot(agn['times'], agn['full_model'],
                            '-', linewidth=2, alpha=0.8,
                            color='red', label='Model')
            axes[i, 3].set_ylabel('Magnitude')
            axes[i, 3].set_title('Observed Data')
            axes[i, 3].invert_yaxis()
            axes[i, 3].legend(fontsize=8)
            axes[i, 3].grid(alpha=0.3)

            # X-labels only on bottom row
            if i == n_examples - 1:
                for j in range(4):
                    axes[i, j].set_xlabel('Time (days)')

        plt.tight_layout()
        return fig

