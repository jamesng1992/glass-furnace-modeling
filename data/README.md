# Data Directory

Place your data files here. Expected files:

| File | Description |
|------|-------------|
| `LQI_data.xlsx` | LQI Excel workbook with **TrueStates** and **EstimatedStates** sheets |
| `JobData.csv` | *(Optional)* Operational production data |
| `Consumption 23-24-25.xlsx` | *(Optional)* Daily consumption by furnace type |

## LQI Data Format

The Excel file should contain two sheets:

### TrueStates / EstimatedStates

| Column | Description | Unit |
|--------|-------------|------|
| `t` or `time` | Time | hours |
| `u1` or `Charging rate` | Charging rate | m³/h |
| `u2` or `Pull rate` | Pull rate | m³/h |
| `w` | Water disturbance | m³/h |
| `h` or `Level` | Glass level | m |
| `v` or `Level rate` | Level change rate | m/h |
| `q_m` or `Melting rate` | Melting rate | m³/h |
| `z1`–`z4` | Transport delay chain states | m³/h |

> **Note:** Column names are automatically standardised by the data loader.
