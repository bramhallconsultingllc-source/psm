# Predictive Staffing Model (PSM)

This repo contains a staffing calculator that converts average daily visits into recommended daily staffing.

## Core Logic
- Uses a staffing ratios table (`data/staffing_ratios.csv`)
- Applies **linear interpolation** for visits between rows
- Rounds all roles **UP** to the nearest 0.25 for conservative staffing
- XRT is fixed at 1.0

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
