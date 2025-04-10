import ast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn.utils.rnn import pad_sequence 
import uproot

"""Memento: features = ["id", "status", "px", "py", "pz", "e", "m"]

What kind of processing?

id is a categorical input. Embedding learnable technique
might be more efficient than one-hot-encoding.

status does not need to be processed: it is 23 for
particles in data_23 dataset, and whatever it needs to
be for particles in data_final dataset.

px, py and pz are continuous variables, hence why
standardization is the most appropriate processing
method.

e and m are continuous variables too, but unlike
px, py and pz they can not be negative. Log-scaling
is the most appropriate normalization method.
"""

with uproot.open("events.root") as file:
    df_23 = file["tree_23"].arrays(library="pd")
    df_final = file["tree_final"].arrays(library="pd")

# Dropping repetitive columns.
df_23 = df_23.drop(columns=[
    "nstatus_23", "npx_23", "npy_23", "npz_23", "ne_23", "nm_23"
    ])

# Adding the event ID
# (useful to rebuild the dataframe after the explosion).
df_23["event_number"] = [ii + 1 for ii in range(len(df_23))]

"""The problem at hand regards the fact that columns are made
by arrays, hence why it is not possible to standardize the df
using StandardScaler() or such. In order to do it anyway, 
the dataframe needs to explode, so that every row has only one value.
In order to make the dataframe explode, the entries of the dataframe
are converted to lists. 
"""

def convert_to_list(value):
    """
    Function aimed at converting strings into lists.

    Args:
        value (any): the input value, can be any type.

    Return:
        value: it is the value in input with a list type if the type of
        the input is str, otherwise it is the input value unchanged.
    """
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)  # ast.literal_eval raises an
                                            # exception if value is not
                                            # a valid datatype.
        except (ValueError, SyntaxError):
            return value
    return value

# Converting every column.
for col in [
    "id_23", "status_23", "px_23", "py_23",
    "pz_23", "e_23", "m_23", "event_number"
    ]:
    df_23[col] = df_23[col].apply(convert_to_list)

# Exploding the dataframe.
df_23_exploded = df_23.explode(
    ["id_23", "status_23", "px_23", "py_23", "pz_23", "e_23", "m_23"],
    ignore_index=True
    )

print(df_23_exploded)

# ===== Standardization of px_23, py_23 and pz_23. =====
df_23_exploded[["px_23", "py_23", "pz_23"]] = StandardScaler().fit_transform(
    df_23_exploded[["px_23", "py_23", "pz_23"]]
)

# ===== Log-scaling of e_23 and m_23. =====
# In order to apply np.log1p(), the entries of the dataframe need to
# be converted to floats.
df_23_exploded["e_23"] = pd.to_numeric(
    df_23_exploded["e_23"], errors="coerce")
df_23_exploded["m_23"] = pd.to_numeric(
    df_23_exploded["m_23"], errors="coerce")

df_23_exploded[["e_23", "m_23"]] = df_23_exploded[
    ["e_23", "m_23"]].apply(np.log1p)

"""Once the normalization process is finished, the dataframe
is reconstitued with its initial shape.
"""
df_23_stand = (df_23_exploded.groupby(["event_number"]).agg({
    "nid_23": 'min', "id_23": list, "status_23": list, "px_23": list, "py_23": list,
    "pz_23": list, "e_23": list, "m_23": list
}).reset_index())

df_23_stand = df_23_stand.drop(columns=["status_23", "event_number"])

"""Once the original division per event is retrieved, the dataframe
needs to be converted to a Torch tensor readable by the transformer.
"""
events_23 = []
for _, row in df_23_stand.iterrows():
    num_particles = row["nid_23"]
    event = []
    for ii in range(num_particles):
        particle = [
            row["id_23"][ii],
            row["px_23"][ii],
            row["py_23"][ii],
            row["px_23"][ii],
            row["e_23"][ii],
            row["m_23"][ii]
        ]
        event.append(particle)
    events_23.append(event)

event_23_tensors = [
    torch.tensor(event, dtype = torch.float32) for event in events_23
    ]

"""Padding is necessary since every event has a different
number of particles.
"""
padded_23 = pad_sequence(
    event_23_tensors, batch_first=True, padding_value=0.0
    )

"""The padded sequence needs to be discriminated: actual particles
vs padding. In order to do so, an attention_mask is implemented.
"""
attention_mask = torch.tensor(
    [[1]*len(event) 
    + [0]*(padded_23.shape[1] 
    - len(event)) for event in events_23],
    dtype=torch.bool
)
