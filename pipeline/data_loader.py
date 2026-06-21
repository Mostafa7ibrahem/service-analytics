import pandas as pd
import io
from typing import Dict

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        rename[c] = c.strip().lower().replace(" ", "_").replace("-", "_")
    df = df.rename(columns=rename)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

DTYPES = {
    "id": "int64", "user_id": "int64", "order_id": "int64",
    "experience_years": "int64", "completed_jobs": "int64",
    "hourly_rate": "float64", "minimum_charge": "float64",
    "commission": "float64", "stars": "int64",
    "latitude": "float64", "longitude": "float64",
}
DATES = {"created_at", "accepted_at", "completed_at", "cancelled_at",
         "date_joined", "last_login", "profile_created_at"}

def load_csv(content: bytes, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception:
        return pd.DataFrame()
    df = _normalize(df)
    for col, dt in DTYPES.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dt)
            except Exception:
                pass
    for d in DATES:
        if d in df.columns:
            df[d] = pd.to_datetime(df[d], errors="coerce")
    if name == "orders":
        df = df.loc[:, ~df.columns.str.startswith("unnamed")]
    return df

def load_all(files: Dict[str, bytes]) -> Dict[str, pd.DataFrame]:
    return {n: load_csv(c, n) for n, c in files.items()}
