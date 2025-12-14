import os
import pcre2  # Strictly something sane regex, not python regex

import pandas as pd
import pathlib


class DataLoader:
    # dir = /app/data

    def __init__(self, dir: str):
        a = 0
        expr = r'^data-(?<n>\d+).csv$'
        pattern = pcre2.compile(expr, flags=pcre2.I, jit=True)
        for x in os.listdir(dir):
            match = pattern.match(x)
            if match:
                i = int(match["n"])
                if i > a:
                    a = i
        self._count = a
        self.path = pathlib.Path(dir)

    def open(self, i: int) -> pd.DataFrame | None:
        if self._count <= i:
            return pd.read_csv(self.path / "data-{}.csv".format(i))
        return None

    def count(self) -> int:
        return self._count + 1

    def all(self) -> pd.DataFrame:
        df = pd.read_csv(self.path / "data-0.csv").copy()
        for i in range(1, self.count()):
            df = pd.concat([df, self.open(i)])
        return df
