#multispec/parameters.py
import numpy as np
from asteval import Interpreter
from typing import Any

class FitParameter:
    """
    Reads in fit parameters from input file
    """
    def __init__(self, name: str, setup_dict: dict) -> None:
        self.name = name

        self.value = setup_dict.get('value', 0.0)
        self.min_val = setup_dict.get('min', -float('inf'))
        self.max_val = setup_dict.get('max', float('inf'))
        self.vary = setup_dict.get('vary', True)
        self.expr = setup_dict.get('expr', None)
        if self.expr is not None:
            self.vary = False

    def get_value(self, namespace: dict | None = None, evaluator: Any = None) -> float | np.ndarray:
        if self.expr:
            try:
                if evaluator is None:
                    evaluator = Interpreter()                
                safe_expr = self.expr.replace('.', '_')
                for key, val in namespace.items():
                    evaluator.symtable[key] = val
                return evaluator(safe_expr)
            except Exception as e:
                name = getattr(self, 'name', 'Array') 
                raise RuntimeError(f"Failed to evaluate expression '{self.expr}' for {name}: {e}")
                
        return self.value

    def set_value(self, new_value: float) -> None:
        self.value = max(self.min_val, min(self.max_val, new_value))

    def __repr__(self) -> str:
        state = f"expr='{self.expr}'" if self.expr else f"vary={self.vary}"
        return f"<FitParameter '{self.name}': value={self.value:.4f}, {state}>"

class ArrayParameter:
    "Reads in array of amplitudes from input file, separate class than scalars"
    def __init__(self, name: str, array_dict: dict, expected_keys: list[str] | None = None, field_indices: list[int] | None = None) -> None:
        self.name = name
        self.expr = array_dict.get('expr', None)

        if self.expr is not None:
            self.vary = False
            self.value = None
            self.min_val = None
            self.max_val = None
        else:
            flat_values = []
            self.row_keys = []
            self.row_sizes = []
            toml_key_map: dict[str, str] = {}
            for k in array_dict:
                if k in ('min', 'max', 'expr'):
                    continue
                toml_key_map[k] = k
                try:
                    val = float(k)
                    toml_key_map[f"{val:g}"] = k      
                    toml_key_map[f"{val:.1f}"] = k     
                    toml_key_map[str(val)] = k
                except ValueError:
                    pass

            keys_to_load = expected_keys if expected_keys else [k for k in array_dict if k not in ('min', 'max', 'expr')]

            for exp_key in keys_to_load:
                matched_key = toml_key_map.get(str(exp_key))

                if matched_key is None:
                    try:
                        target_f = float(exp_key)
                        for actual_k in array_dict:
                            if actual_k in ('min', 'max', 'expr'):
                                continue
                            try:
                                if np.isclose(float(actual_k), target_f, atol=1e-3):
                                    matched_key = actual_k
                                    break
                            except ValueError:
                                continue
                    except ValueError:
                        pass

                if matched_key is None:
                    avail = [k for k in array_dict if k not in ('min', 'max', 'expr')]
                    raise KeyError(
                        f"In band '{self.name}': Missing amplitude array for temperature '{exp_key}'. "
                        f"Found keys in TOML: {avail}"
                    )

                values = array_dict[matched_key]
                if isinstance(values, (int, float)):
                    values = [values]

                if field_indices is not None and len(values) > len(field_indices):
                    values = [values[idx] for idx in field_indices]

                self.row_keys.append(matched_key)
                self.row_sizes.append(len(values))
                flat_values.extend(values)

            if len(flat_values) == 0:
                raise ValueError(f"No amplitude data was loaded for parameter '{self.name}'.")
               
            self.value = np.array(flat_values, dtype=float)
            self.vary = True
            self.min_val = array_dict.get('min', -np.inf)
            self.max_val = array_dict.get('max', np.inf)

    def get_value(self, namespace: dict | None = None, evaluator: Any = None) -> np.ndarray | float:
        if self.expr:
            try:
                if evaluator is None:
                    from asteval import Interpreter
                    evaluator = Interpreter()                
                safe_expr = self.expr.replace('.', '_')
                for key, val in namespace.items():
                    evaluator.symtable[key] = val
                return evaluator(safe_expr)
            except Exception as e:
                name = getattr(self, 'name', 'Array') 
                raise RuntimeError(f"Failed to evaluate expression '{self.expr}' for {name}: {e}")
                        
        return self.value

    def set_value(self, new_array: np.ndarray) -> None:
        if self.expr is not None:
            raise ValueError(f"Cannot manually set '{self.name}'; constrained by {self.expr}")
            
        self.value = np.clip(new_array, self.min_val, self.max_val)