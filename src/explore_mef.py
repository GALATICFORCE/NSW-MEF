from pathlib import Path
import nemed
import aemo_archive_fix

aemo_archive_fix.apply()

CACHE = Path("data/raw/nemed_cache")
CACHE.mkdir(parents= True,exist_ok= True)

df = nemed.get_marginal_emissions(
    start_time = "2025/07/01 00:00",
    end_time = "2025/07/08 00:00",
    cache = str(CACHE),
)

print("shape:", df.shape)
print(df.dtypes)
print(df.head(10))
print(df["Region"].value_counts())

nsw = df[df["Region"] == "NSW1"].copy()
print("NSW1 range:", nsw["Time"].min(), "->", nsw["Time"].max(), "rows:", len(nsw))
print(nsw["Intensity_Index"].describe())
print(nsw["CO2E_ENERGY_SOURCE"].value_counts())

nsw.to_csv("data/raw/mef_nsw1_5min_2025-07-01_7d.csv", index=False)
