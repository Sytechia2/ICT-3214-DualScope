"""Hourly event counts and model inputs for the fusion model (docs/supervised_fusion.md).

Shared by the validation and final-test scripts and by the end-to-end pipeline,
so every path computes the fusion inputs the same way. ``is_machine`` is the
evaluation's machine-account rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

COUNTS = [
    "n_events", "n_failures", "n_sources", "n_destinations",
    "n_new_user_source", "n_new_host_connection", "n_new_user_destination",
    "n_ntlm", "n_network_logon", "n_logon",
]
INPUTS = ["gru_max_event", *COUNTS, "is_machine_account"]


def _codes(column) -> tuple[np.ndarray, np.ndarray]:
    encoded = column.combine_chunks().dictionary_encode()
    return encoded.indices.to_numpy(zero_copy_only=False), encoded.dictionary.to_numpy(zero_copy_only=False)


def hourly_counts(dataset: ds.Dataset, day: int) -> pd.DataFrame:
    """One row per (user, hour) of ``day`` with the fusion count inputs and the ``either`` rule count."""
    table = dataset.to_table(
        columns=[
            "acting_user", "timestamp", "source_computer", "destination_computer", "authentication_type",
            "logon_type", "authentication_orientation", "authentication_result", "is_new_user_source",
            "is_new_host_connection", "is_new_user_destination", "is_machine_account",
        ],
        filter=ds.field("dataset_day") == day,
    )
    user, users = _codes(table["acting_user"])
    frame = pd.DataFrame({
        "u": user,
        "hour": 1 + ((table["timestamp"].to_numpy() - 1) // 3600) * 3600,
        "src": _codes(table["source_computer"])[0],
        "dst": _codes(table["destination_computer"])[0],
        "n_failures": pc.equal(table["authentication_result"], "Fail").to_numpy(),
        "n_new_user_source": table["is_new_user_source"].to_numpy(),
        "n_new_host_connection": table["is_new_host_connection"].to_numpy(),
        "n_new_user_destination": table["is_new_user_destination"].to_numpy(),
        "n_ntlm": pc.equal(table["authentication_type"], "NTLM").to_numpy(),
        "n_network_logon": pc.equal(table["logon_type"], "Network").to_numpy(),
        "n_logon": pc.equal(table["authentication_orientation"], "LogOn").to_numpy(),
        "is_machine_account": table["is_machine_account"].to_numpy(),
    })
    del table
    frame["either"] = frame["n_new_user_source"] | frame["n_new_host_connection"]
    flags = ["n_failures", "n_new_user_source", "n_new_host_connection", "n_new_user_destination", "n_ntlm", "n_network_logon", "n_logon", "either"]
    grouped = frame.groupby(["u", "hour"], sort=False)
    out = grouped[flags].sum()
    out["n_events"] = grouped.size()
    out["is_machine_account"] = grouped["is_machine_account"].max()
    for column, name in (("src", "n_sources"), ("dst", "n_destinations")):
        out[name] = frame[["u", "hour", column]].drop_duplicates().groupby(["u", "hour"], sort=False).size()
    out = out.reset_index()
    out.insert(0, "user", users[out.pop("u").to_numpy()])
    out["day"] = day
    return out


def model_inputs(frame: pd.DataFrame) -> np.ndarray:
    """The fusion model's input matrix, columns in ``INPUTS`` order."""
    return frame[INPUTS].to_numpy(np.float64)


def is_machine(users: np.ndarray) -> np.ndarray:
    """True for machine accounts: the part before ``@`` ends with ``$``."""
    return np.fromiter((str(u).split("@", 1)[0].endswith("$") for u in users), dtype=bool, count=len(users))
