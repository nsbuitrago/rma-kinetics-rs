import polars as pl

def rlu_to_nm(df: pl.DataFrame, rlu_col: str = "rlu", reporter: str = "rma") -> pl.DataFrame:
    """
    Convert RLU values to nM concentration

    Arguments
    ---------
    df : polars.DataFrame
        DataFrame containing the RLU values.
    rlu_col : str, optional
        Column name containing the RLU values, by default "gluc".
    reporter : str, optional
        Reporter name, by default "rma".

    Returns
    -------
    polars.DataFrame
        DataFrame containing the nM concentration values in the 'concentration' column.
    """

    if reporter not in ["rma", "gluc"]:
        raise ValueError(f"Reporter {reporter} not supported. Must be 'rma' or 'gluc'.")

    if reporter == "rma":
        return df.with_columns(((pl.col(rlu_col) * 0.0012 + 26.96) / 44).alias("concentration"))
    elif reporter == "gluc":
        return df.with_columns((pl.col(rlu_col) * 168669125 / 227.2).alias("concentration"))
    else:
        raise ValueError(f"Reporter {reporter} not supported. Must be 'rma' or 'gluc'.")