# NSW average emissions intensity (AEI): method note

- **What it is.** For each 5-minute interval, the total greenhouse emissions (tonnes CO2-equivalent) of New South Wales power stations divided by the electricity they sent into the grid (MWh). Unit: tCO2e/MWh, which is the same as kg per kWh.
- **Data.** AEMO's 5-minute output readings for each generating unit (DISPATCH_UNIT_SCADA) and AEMO's emission factor for each unit (GENUNITS, version as at 1 July 2026, the same factors used for our MEF labels). We processed them with the open-source NEMED tool.
- **Energy.** A unit's output over 5 minutes is the average of its readings at the start and end of the interval. We then subtract the power station's own electricity use to get "sent-out" energy, i.e. what actually reaches the grid.
- **Scope.** Only generators located in NSW (region NSW1). Batteries and other two-way units are not included, and neither is electricity imported from other states.
- **Excluded units.** Three units have no emission factor in AEMO's data: Colongra CG2 and CG4, and ERGT01. They ran in 1,536 of the 105,120 intervals and supplied 0.023% of NSW sent-out energy. They are left out of both the emissions and the energy totals.
- **Why we do not use NEMED's ready-made intensity.** NEMED's `Intensity_Index` divides sent-out emissions by energy *before* the stations' own use is removed. This mixes two bases and understates intensity by about 4–5%. We therefore compute AEI ourselves as total emissions ÷ sent-out energy.
- **Timestamps.** `TimeEnding` is the end of each 5-minute interval: the 00:05 row covers 00:00–00:05. This is the same convention as the MEF data. Times are NEM time (AEST, UTC+10, no daylight saving).
- **Coverage.** 2025-07-01 00:05 to 2026-07-01 00:00: 105,120 intervals, no gaps, no missing values.
- **Columns.**
  - `TimeEnding`
  - `Energy_SO`: sent-out energy in the interval, MWh
  - `Total_Emissions`: tCO2e in the interval
  - `aei`: `Total_Emissions` ÷ `Energy_SO`, tCO2e/MWh
- **Hourly values.**
  - A 5-minute row belongs to the hour in which its interval starts (`TimeEnding` minus 5 minutes).
  - The hourly AEI is the simple average of the twelve 5-minute `aei` values in that hour, left blank if fewer than 6 are available. This matches a load charging at constant power, and it is the same rule used for the MEF and price tables.
- **Cross-check.** The hourly value can also be computed as the hour's total emissions ÷ total sent-out energy. The quality report states the largest difference between the two methods.
