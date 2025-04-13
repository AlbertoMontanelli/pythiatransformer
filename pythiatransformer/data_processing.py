"""In this code the data from pythia_generator.py is processed.
First, the ROOT files are imported as pandas dataframes.
Then, the features are standardized.
Last, the dataframes are converted to Torch tensors and split
between training, validation and test sets.
"""
import ast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn.utils.rnn import pad_sequence
import uproot


with uproot.open("events.root") as file:
    df_23 = file["tree_23"].arrays(library="pd")
    df_final = file["tree_final"].arrays(library="pd")

def convert_to_list(value):
    """
    Function aimed at converting strings into lists. Helper function.

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

def preprocess_dataframe(df, event_particles_col = "nid_23",
                         id_col="id_23", status_col="status_23",
                         px_col="px_23", py_col="py_23", pz_col="pz_23",
                         e_col="e_23", m_col="m_23"):
    """Function that preprocesses the DataFrame, dropping repetitive
    columns, exploding the dataframe, doing normalization, and
    re-building its original structure. The function is initialized
    with respect to df_23.

    Args:
        df (pandas DataFrame): DataFrame.
        event_particles_col (str): name of the event particles column.
        id_col (str): name of the particle ID column.
        status_col (str): name of the particle status column.
        px_col (str): name of the particle px column.
        py_col (str): name of the particle py column.
        pz_col (str): name of the particle pz column.
        e_col (str): name of the particle energy column.
        m_col (str): name of the particle mass column.

    Return:
        df_stand (pandas DataFrame): standardized DataFrame.
    """
    
    # Dropping repetitive columns.
    drop_cols = [f"n{col}" for col in [
        status_col, px_col, py_col, pz_col, e_col, m_col
    ]]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Adding the event ID
    # (useful to rebuild the dataframe after the explosion).
    df["event_number"] = [ii + 1 for ii in range(len(df))]

    """The problem at hand regards the fact that columns are made
    by arrays, hence why it is not possible to standardize the df
    using StandardScaler() or such. In order to do it anyway, the
    dataframe needs to explode, so that every row has only one value.
    """
    # Converting to lists.
    for col in [id_col, status_col, px_col, py_col, pz_col, e_col, m_col]:
        df[col] = df[col].apply(convert_to_list)

    # Exploding the dataframe.
    df_exploded = df.explode(
        [id_col, status_col, px_col, py_col, pz_col, e_col, m_col],
        ignore_index=True
    )

    # ===== Standardization of px, py and pz. =====
    df_exploded[[px_col, py_col, pz_col]] = StandardScaler().fit_transform(
        df_exploded[[px_col, py_col, pz_col]]
    )

    # ===== Log-scaling of e and m. =====
    # In order to apply np.log1p(), the entries of the dataframe need to
    # be converted to floats.
    df_exploded[e_col] = pd.to_numeric(
        df_exploded[e_col], errors="coerce")
    df_exploded[m_col] = pd.to_numeric(
        df_exploded[m_col], errors="coerce")

    df_exploded[[e_col, m_col]] = df_exploded[
        [e_col, m_col]].apply(np.log1p)

    """Once the normalization process is finished, the dataframe
    is reconstitued with its initial shape.
    """
    df_stand = (df_exploded.groupby(["event_number"]).agg({
        event_particles_col: 'min', id_col: list, status_col: list,
        px_col: list, py_col: list, pz_col: list, e_col: list, m_col: list
    }).reset_index())

    df_stand = df_stand.drop(columns=[status_col, "event_number"])

    return df_stand


def dataframe_to_padded_tensor(df_stand, event_particles_col = "nid_23",
                         id_col="id_23", px_col="px_23", py_col="py_23",
                         pz_col="pz_23", e_col="e_23", m_col="m_23"):
    """Function that converts the standardized DataFrame to a Torch
    padded tensor.

    Args:
        df_stand (pandas DataFrame): DataFrame.
        event_particles_col (str): name of the event particles column.
        id_col (str): name of the particle ID column.
        px_col (str): name of the particle px column.
        py_col (str): name of the particle py column.
        pz_col (str): name of the particle pz column.
        e_col (str): name of the particle energy column.
        m_col (str): name of the particle mass column.

    Return:
        padded_tensor: Torch padded tensor, obtained by the conversion
                       of the DataFrame.
        attention_mask: attention mask that considers the padding.
    """
    
    """Once the original division per event is retrieved, the dataframe
    needs to be converted to a Torch tensor readable by the transformer.
    """
    events = []
    for _, row in df_stand.iterrows():
        num_particles = row[event_particles_col]
        event = []
        for ii in range(num_particles):
            particle = [
                row[id_col][ii],
                row[px_col][ii],
                row[py_col][ii],
                row[pz_col][ii],
                row[e_col][ii],
                row[m_col][ii]
            ]
            event.append(particle)
        events.append(event)

    event_tensor = [
        torch.tensor(event, dtype = torch.float32) for event in events
    ]

    """Padding is necessary since every event has a different
    number of particles.
    """
    padded_tensor = pad_sequence(
        event_tensor, batch_first=True, padding_value=0.0
    )

    """The padded sequence needs to be discriminated: actual particles
    vs padding. In order to do so, an attention_mask is implemented.
    """
    attention_mask = torch.tensor(
        [[1]*len(event) 
        + [0]*(padded_tensor.shape[1] 
        - len(event)) for event in events],
        dtype=torch.bool
    )
    
    return padded_tensor, attention_mask

def train_val_test_split(
        tensor, train_perc = 0.6, val_perc = 0.2
        ):
    """Function that splits a tensor in training, validation and test sets.

    Args:
        tensor (Torch tensor): data in the form of a Torch tensor.
        train_perc (float): fraction of the data used for training.
        val_perc (float): fraction of the data used for validation.

    Return:
        training_set (Torch tensor): training set.
        validation_set (Torch tensor): validation set.
        test_set (Torch tensor): test set.
    """
    nn = len(tensor)
    len_train = int(train_perc*nn)
    len_val = int(val_perc*nn)

    training_set = tensor[:len_train]
    validation_set = tensor[len_train + 1:len_train + len_val]
    test_set = tensor[len_train + len_val + 1:]

    return training_set, validation_set, test_set


df_23_stand = preprocess_dataframe(df_23)
padded_tensor_23, attention_mask_23 = dataframe_to_padded_tensor(df_23_stand)

df_final_stand = preprocess_dataframe(df_final, "nid_final", "id_final",
                                      "status_final", "px_final", "py_final",
                                      "pz_final", "e_final", "m_final")
padded_tensor_final, attention_mask_final = dataframe_to_padded_tensor(
    df_final_stand, "nid_final", "id_final", "px_final", "py_final",
    "pz_final", "e_final", "m_final"
)

training_set_23, validation_set_23, test_set_23 = train_val_test_split(padded_tensor_23)
training_set_final, validation_set_final, test_set_final = train_val_test_split(padded_tensor_final)
