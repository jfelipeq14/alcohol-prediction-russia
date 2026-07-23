import pandas as pd
from pathlib import Path


class CSVLoader:
    def __init__(self, filepath: Path):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                df = pd.read_csv(self.filepath, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            df = pd.read_csv(self.filepath, encoding="latin-1", encoding_errors="replace")

        df["Type of alcoholic beverages"] = (
            df["Type of alcoholic beverages"]
            .str.replace(r"[^A-Za-z]ider", "Cider", regex=True)
        )
        return df
