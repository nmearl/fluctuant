import numpy as np
import pandas as pd
import warnings


class AstromerCatalogProcessor:
    """Process a real light curve catalog through ASTROMER to produce embeddings.

    Example usage::

        processor = AstromerCatalogProcessor()
        embeddings, ids = processor.process_catalog(
            catalog_path='my_agn_catalog.csv',
            time_col='mjd',
            mag_col='mag',
            err_col='magerr',
            id_col='object_id',
        )
    """

    def __init__(self, pretrained_weights='macho', max_obs=200):
        from ASTROMER.models import SingleBandEncoder
        self.max_obs = max_obs
        self.model = SingleBandEncoder()
        print(f"Loading pretrained weights: {pretrained_weights}")
        self.model = self.model.from_pretraining(pretrained_weights)

    def load_catalog(self, catalog_path, format='auto'):
        """
        Load a catalog file.

        Parameters
        ----------
        catalog_path : str
            Path to the catalog file.
        format : {'auto', 'csv', 'parquet', 'fits'}
            File format; 'auto' detects from the file extension.
        """
        if format == 'auto':
            if catalog_path.endswith('.csv'):
                return pd.read_csv(catalog_path)
            elif catalog_path.endswith('.parquet'):
                return pd.read_parquet(catalog_path)
            elif catalog_path.endswith('.fits'):
                from astropy.table import Table
                return Table.read(catalog_path).to_pandas()
            else:
                raise ValueError(f"Cannot auto-detect format for {catalog_path}")

        elif format == 'csv':
            return pd.read_csv(catalog_path)
        elif format == 'parquet':
            return pd.read_parquet(catalog_path)
        elif format == 'fits':
            from astropy.table import Table
            return Table.read(catalog_path).to_pandas()
        else:
            raise ValueError(f"Unknown format: {format}")

    def prepare_lightcurve_data(self, catalog, time_col, mag_col, err_col,
                                id_col=None, band_col=None, band=None):
        """
        Convert a catalog DataFrame to ASTROMER-compatible light curve arrays.

        Parameters
        ----------
        catalog : pd.DataFrame
        time_col : str
            Column name for observation times (MJD).
        mag_col : str
            Column name for magnitudes.
        err_col : str
            Column name for magnitude errors.
        id_col : str, optional
            Column name for object IDs. If None, treats the whole catalog as
            a single object.
        band_col : str, optional
            Column name for the filter/band.
        band : str, optional
            Specific band to select when ``band_col`` is provided.

        Returns
        -------
        list of ndarray
            Each array has shape ``(n_obs, 3)``: [time, mag, err].
        list
            Object IDs corresponding to each light curve.
        """
        # Filter by band if specified
        if band_col is not None and band is not None:
            catalog = catalog[catalog[band_col] == band].copy()
            print(f"Filtered to band '{band}': {len(catalog)} observations")

        # If no ID column, treat as single light curve
        if id_col is None:
            print("No ID column specified - treating as single light curve")
            lc_data = np.column_stack([
                catalog[time_col].values,
                catalog[mag_col].values,
                catalog[err_col].values
            ])
            return [lc_data], ['single_object']

        # Group by object ID
        lightcurves = []
        object_ids = []

        grouped = catalog.groupby(id_col)
        print(f"Found {len(grouped)} unique objects")

        for obj_id, group in grouped:
            # Sort by time
            group = group.sort_values(time_col)

            # Extract data
            times = group[time_col].values
            mags = group[mag_col].values
            errs = group[err_col].values

            # Quality checks
            if len(times) < 5:
                warnings.warn(f"Object {obj_id} has only {len(times)} observations - skipping")
                continue

            if np.any(~np.isfinite(times)) or np.any(~np.isfinite(mags)):
                warnings.warn(f"Object {obj_id} has NaN/Inf values - skipping")
                continue

            # Create array
            lc_data = np.column_stack([times, mags, errs])

            lightcurves.append(lc_data)
            object_ids.append(obj_id)

        print(f"Successfully prepared {len(lightcurves)} light curves")
        return lightcurves, object_ids

    def preprocess_for_astromer(self, lightcurves):
        """Mean-center times and magnitudes; leave errors unchanged."""
        processed = []

        for lc in lightcurves:
            times = lc[:, 0] - lc[:, 0].mean()
            mags = lc[:, 1] - lc[:, 1].mean()
            errs = lc[:, 2]

            processed.append(np.column_stack([times, mags, errs]))

        return processed

    def window_long_lightcurves(self, lightcurves, object_ids):
        """
        Split light curves longer than max_obs into windows.
        Returns windowed light curves and updated object IDs.
        """
        windowed_lcs = []
        windowed_ids = []

        for lc, obj_id in zip(lightcurves, object_ids):
            if len(lc) <= self.max_obs:
                windowed_lcs.append(lc)
                windowed_ids.append(obj_id)
            else:
                # Split into windows
                n_windows = len(lc) // self.max_obs
                for i in range(n_windows):
                    start = i * self.max_obs
                    end = start + self.max_obs
                    windowed_lcs.append(lc[start:end])
                    windowed_ids.append(f"{obj_id}_window{i}")

                # Handle remainder
                if len(lc) % self.max_obs > 50:  # Only if remainder is substantial
                    windowed_lcs.append(lc[-self.max_obs:])
                    windowed_ids.append(f"{obj_id}_window{n_windows}")

        return windowed_lcs, windowed_ids

    def generate_embeddings(self, lightcurves, pool='mean'):
        """
        Encode light curves with the ASTROMER encoder.

        Parameters
        ----------
        lightcurves : list of ndarray, shape (n_obs, 3)
        pool : {'mean', 'max', 'last', 'none'}
            How to collapse the temporal dimension.

        Returns
        -------
        ndarray, shape (n_lightcurves, d_model)
            Pooled embeddings (``d_model=256`` for MACHO weights), or a list
            of per-observation arrays when ``pool='none'``.
        """
        print("Generating embeddings...")
        raw_embeddings = self.model.encode(lightcurves)

        if pool == 'none':
            return raw_embeddings

        # Pool over time dimension
        pooled = []
        for emb in raw_embeddings:
            if pool == 'mean':
                pooled.append(emb.mean(axis=0))
            elif pool == 'max':
                pooled.append(emb.max(axis=0))
            elif pool == 'last':
                pooled.append(emb[-1])
            else:
                raise ValueError(f"Unknown pooling method: {pool}")

        return np.vstack(pooled)

    def process_catalog(self, catalog_path, time_col, mag_col, err_col,
                        id_col=None, band_col=None, band=None,
                        format='auto', pool='mean', save_path=None):
        """
        End-to-end pipeline: load catalog → preprocess → embed.

        Parameters
        ----------
        catalog_path : str
        time_col : str
        mag_col : str
        err_col : str
        id_col : str, optional
        band_col : str, optional
        band : str, optional
        format : {'auto', 'csv', 'parquet', 'fits'}
        pool : {'mean', 'max', 'last', 'none'}
        save_path : str, optional
            If provided, save embeddings to this path (.npz, .csv, or .parquet).

        Returns
        -------
        embeddings : ndarray, shape (n_objects, d_model)
        object_ids : list
        """
        # Load catalog
        print(f"Loading catalog from {catalog_path}")
        catalog = self.load_catalog(catalog_path, format=format)
        print(f"Catalog shape: {catalog.shape}")

        # Prepare light curves
        lightcurves, object_ids = self.prepare_lightcurve_data(
            catalog, time_col, mag_col, err_col, id_col, band_col, band
        )

        # Preprocess (normalize)
        lightcurves = self.preprocess_for_astromer(lightcurves)

        # Window long light curves
        lightcurves, object_ids = self.window_long_lightcurves(lightcurves, object_ids)

        # Generate embeddings
        embeddings = self.generate_embeddings(lightcurves, pool=pool)

        # Save if requested
        if save_path:
            self.save_embeddings(embeddings, object_ids, save_path)

        return embeddings, object_ids

    def save_embeddings(self, embeddings, object_ids, save_path):
        """Save embeddings to file."""
        if save_path.endswith('.npz'):
            np.savez(save_path, embeddings=embeddings, ids=object_ids)
        elif save_path.endswith('.csv'):
            df = pd.DataFrame(embeddings)
            df.insert(0, 'object_id', object_ids)
            df.to_csv(save_path, index=False)
        elif save_path.endswith('.parquet'):
            df = pd.DataFrame(embeddings)
            df.insert(0, 'object_id', object_ids)
            df.to_parquet(save_path, index=False)
        else:
            raise ValueError(f"Unknown save format: {save_path}")

        print(f"Embeddings saved to {save_path}")