import pandas as pd
ATT=pd.read_csv("attribution.csv"); CF=pd.read_csv("counterfactuals.csv"); S=pd.read_csv("sensitivity.csv")
exec(open("xai_calib.py").read().split("# ---------- figure")[1])
