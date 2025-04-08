import ast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
            return ast.literal_eval(value)  # Converting the list to a string.
        except (ValueError, SyntaxError):
            return value  # If not a list, returns the original value.
    return value

# Converting every column.
for col in [
    "id_23", "status_23", "px_23", "py_23"
    "pz_23", "e_23", "m_23", "event_number"
    ]:
    df_23[col] = df_23[col].apply(convert_to_list)

# Exploding the dataframe.
df_23_exploded = df_23.explode(
    ["id_23", "status_23", "px_23", "py_23", "pz_23", "e_23", "m_23"],
    ignore_index=True
    )

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
