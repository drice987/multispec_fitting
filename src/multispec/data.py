import re
from pathlib import Path

import numpy as np


class DataSet:
    """
    Loads experimental data, aligns it with the requested TOML conditions,
    and flattens it for the least-squares minimizer.
    """

    def __init__(
        self,
        dataset_config: dict,
        select_fields: list[float] | None = None,
        select_temps: list[float] | None = None,
    ) -> None:
        self.filename = Path(dataset_config["filename"])
        self.temp_tolerance = dataset_config.get("temp_tolerance", 0.5)
        self.is_vtvh = "fields" in dataset_config and "temperatures" in dataset_config

        self.x = None
        self.y_matrix = None
        self.y_flat = None
        self.real_temperatures = []
        self.real_fields = []
        self.matched_headers = []
        self.conditions = []

        if self.is_vtvh:
            self.master_fields = [float(f) for f in dataset_config["fields"]]
            self.master_temperatures = [float(t) for t in dataset_config["temperatures"]]
            # Fields
            if select_fields is not None:
                self.active_field_indices = [
                    i
                    for i, f in enumerate(self.master_fields)
                    if any(np.isclose(f, sf, atol=1e-2) for sf in select_fields)
                ]
                if not self.active_field_indices:
                    raise ValueError(
                        f"None of the requested fields {select_fields} match the dataset fields: {self.master_fields}"
                    )
                self.toml_fields = [self.master_fields[i] for i in self.active_field_indices]
            else:
                self.active_field_indices = list(range(len(self.master_fields)))
                self.toml_fields = list(self.master_fields)

            # Temperatures
            if select_temps is not None:
                self.active_temp_indices = [
                    i
                    for i, t in enumerate(self.master_temperatures)
                    if any(np.isclose(t, st, atol=1e-2) for st in select_temps)
                ]
                if not self.active_temp_indices:
                    raise ValueError(
                        f"None of the requested temperatures {select_temps} match the dataset temperatures: {self.master_temperatures}"
                    )
                self.toml_temperatures = [
                    self.master_temperatures[i] for i in self.active_temp_indices
                ]
            else:
                self.active_temp_indices = list(range(len(self.master_temperatures)))
                self.toml_temperatures = list(self.master_temperatures)

            self.toml_columns = []
        else:
            self.master_fields = []
            self.master_temperatures = []
            self.active_field_indices = None
            self.active_temp_indices = None
            self.toml_fields = []
            self.toml_temperatures = []
            self.toml_columns = dataset_config.get("columns", [])

        self._load_and_flatten()

    def _load_and_flatten(self) -> None:
        if not self.filename.exists():
            raise FileNotFoundError(f"Data file not found: {self.filename}")

        with open(self.filename) as f:
            header_line = f.readline().strip()
        headers = header_line.split()
        raw_data = np.loadtxt(self.filename, skiprows=1)
        self.x = raw_data[:, 0]

        selected_columns = []

        if self.is_vtvh:
            header_pattern = re.compile(r"^([0-9]*\.?[0-9]+)K_([0-9]*\.?[0-9]+)T$")
            parsed_headers = []
            for idx, h in enumerate(headers):
                m = header_pattern.match(h)
                if m:
                    parsed_headers.append(
                        {"T": float(m.group(1)), "B": float(m.group(2)), "idx": idx, "header": h}
                    )

            for set_t in self.toml_temperatures:
                for set_b in self.toml_fields:
                    matches = [
                        p
                        for p in parsed_headers
                        if np.isclose(p["B"], float(set_b))
                        and abs(p["T"] - float(set_t)) <= self.temp_tolerance
                    ]
                    if not matches:
                        raise ValueError(
                            f"Could not find a column for {set_t}K_{set_b}T within +/- {self.temp_tolerance}K window."
                        )
                    best_match = min(matches, key=lambda x: abs(x["T"] - float(set_t)))

                    selected_columns.append(best_match["idx"])
                    self.real_temperatures.append(best_match["T"])
                    self.real_fields.append(best_match["B"])
                    self.matched_headers.append(best_match["header"])
                    self.conditions.append(f"{set_t}K_{set_b}T")
        else:
            if not self.toml_columns:
                self.toml_columns = headers[1:]

            for col_name in self.toml_columns:
                if col_name not in headers:
                    raise ValueError(
                        f"Column '{col_name}' not found in data file headers: {headers}"
                    )
                idx = headers.index(col_name)
                selected_columns.append(idx)
                self.real_temperatures.append(0.0)
                self.real_fields.append(0.0)
                self.matched_headers.append(col_name)
                self.conditions.append(col_name)

        self.y_matrix = raw_data[:, selected_columns]
        self.y_flat = self.y_matrix.T.flatten()

    def get_x(self) -> np.ndarray:
        return self.x

    def get_y_flat(self) -> np.ndarray:
        return self.y_flat
