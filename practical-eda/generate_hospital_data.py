"""
Synthetic Ghana hospital admissions dataset for teaching Exploratory Data Analysis (EDA).

* 10,000 rows x 20 columns, written to ghana_hospital_data.csv
* ALL DATA IS SYNTHETIC. Patients, facilities and values are invented; facility names are generic.
* Deliberately messy: missing values (many placeholder styles), duplicates, outliers, unit mix-ups,
  mixed date formats and misspelled diseases, regions and other labels.

Run:  python generate_hospital_data.py
"""
import json
import numpy as np
import pandas as pd

SEED = 2024
N_TOTAL, N_EXACT, N_KEY = 10_000, 100, 50
N = N_TOTAL - N_EXACT - N_KEY                      # unique patient records
rng = np.random.default_rng(SEED)
issues = {}                                        # counts of injected problems (for the instructor key)

# --------------------------------------------------------------------------- reference data
REGIONS = {  # region: (weight, towns)
    "Greater Accra": (.17, ["Accra", "Tema", "Madina"]), "Ashanti": (.17, ["Kumasi", "Obuasi", "Ejisu"]),
    "Eastern": (.08, ["Koforidua", "Nkawkaw", "Suhum"]), "Central": (.07, ["Cape Coast", "Winneba", "Kasoa"]),
    "Western": (.07, ["Takoradi", "Tarkwa", "Axim"]), "Volta": (.06, ["Ho", "Keta", "Hohoe"]),
    "Northern": (.08, ["Tamale", "Yendi", "Savelugu"]), "Bono": (.04, ["Sunyani", "Berekum", "Dormaa"]),
    "Bono East": (.04, ["Techiman", "Kintampo", "Atebubu"]), "Upper East": (.05, ["Bolgatanga", "Navrongo", "Bawku"]),
    "Upper West": (.03, ["Wa", "Tumu", "Lawra"]), "Western North": (.04, ["Sefwi Wiawso", "Bibiani", "Juaboso"]),
    "Oti": (.03, ["Dambai", "Nkwanta", "Jasikan"]), "North East": (.02, ["Nalerigu", "Walewale", "Gambaga"]),
    "Savannah": (.02, ["Damongo", "Bole", "Salaga"]), "Ahafo": (.02, ["Goaso", "Kenyasi", "Bechem"]),
}
NORTH = {"Northern", "Savannah", "North East", "Upper East", "Upper West"}
COAST = {"Greater Accra", "Central", "Western", "Volta"}
URBAN = {"Greater Accra", "Ashanti"}
TEACHING_REGIONS = {"Greater Accra", "Ashanti", "Northern", "Central"}

DISEASES = ["Malaria", "Typhoid Fever", "Hypertension", "Diabetes Mellitus", "Pneumonia", "Tuberculosis",
            "HIV/AIDS", "Cholera", "Sickle Cell Disease", "Anaemia", "Urinary Tract Infection",
            "Diarrhoeal Disease", "Acute Respiratory Infection", "Meningitis", "Stroke", "Asthma",
            "Hepatitis B", "Malnutrition", "Road Traffic Injury", "Pregnancy Complications"]
D = {d: i for i, d in enumerate(DISEASES)}
AGE_W = {  # relative disease frequency by age group
    "u5":  {"Malaria": 30, "Pneumonia": 14, "Diarrhoeal Disease": 14, "Acute Respiratory Infection": 12, "Anaemia": 8,
            "Sickle Cell Disease": 6, "Malnutrition": 8, "Meningitis": 3, "Typhoid Fever": 2, "Asthma": 2, "Road Traffic Injury": 1},
    "5_17": {"Malaria": 24, "Typhoid Fever": 10, "Acute Respiratory Infection": 10, "Sickle Cell Disease": 9, "Asthma": 7,
             "Pneumonia": 8, "Anaemia": 6, "Road Traffic Injury": 8, "Diarrhoeal Disease": 6, "Urinary Tract Infection": 3,
             "Meningitis": 3, "Hepatitis B": 2, "Tuberculosis": 2},
    "18_39": {"Malaria": 16, "Typhoid Fever": 9, "HIV/AIDS": 7, "Urinary Tract Infection": 8, "Road Traffic Injury": 10,
              "Hypertension": 6, "Anaemia": 6, "Tuberculosis": 5, "Hepatitis B": 4, "Sickle Cell Disease": 3, "Pneumonia": 3,
              "Diabetes Mellitus": 3, "Cholera": 2, "Asthma": 2},
    "40_59": {"Hypertension": 22, "Diabetes Mellitus": 16, "Malaria": 8, "Stroke": 8, "Tuberculosis": 7, "HIV/AIDS": 6,
              "Typhoid Fever": 5, "Urinary Tract Infection": 5, "Road Traffic Injury": 5, "Pneumonia": 4, "Hepatitis B": 4,
              "Anaemia": 3, "Asthma": 2, "Cholera": 2},
    "60p": {"Hypertension": 24, "Diabetes Mellitus": 17, "Stroke": 15, "Pneumonia": 9, "Malaria": 5, "Urinary Tract Infection": 6,
            "Tuberculosis": 5, "Anaemia": 4, "Road Traffic Injury": 3, "HIV/AIDS": 2, "Asthma": 3, "Cholera": 2, "Typhoid Fever": 3},
}
LOS_MEAN = {"Malaria": 3, "Typhoid Fever": 5, "Hypertension": 3, "Diabetes Mellitus": 5, "Pneumonia": 6, "Tuberculosis": 14,
            "HIV/AIDS": 8, "Cholera": 4, "Sickle Cell Disease": 5, "Anaemia": 4, "Urinary Tract Infection": 3,
            "Diarrhoeal Disease": 3, "Acute Respiratory Infection": 3, "Meningitis": 10, "Stroke": 12, "Asthma": 2.5,
            "Hepatitis B": 7, "Malnutrition": 8, "Road Traffic Injury": 9, "Pregnancy Complications": 5}
COST_BASE = {"Malaria": 150, "Typhoid Fever": 300, "Hypertension": 250, "Diabetes Mellitus": 400, "Pneumonia": 450,
             "Tuberculosis": 600, "HIV/AIDS": 500, "Cholera": 400, "Sickle Cell Disease": 700, "Anaemia": 350,
             "Urinary Tract Infection": 180, "Diarrhoeal Disease": 150, "Acute Respiratory Infection": 120, "Meningitis": 1800,
             "Stroke": 2800, "Asthma": 200, "Hepatitis B": 650, "Malnutrition": 300, "Road Traffic Injury": 1500,
             "Pregnancy Complications": 1600}
MORTALITY = {"Stroke": .12, "Meningitis": .14, "Tuberculosis": .08, "HIV/AIDS": .09, "Cholera": .06, "Road Traffic Injury": .07,
             "Malnutrition": .06, "Pregnancy Complications": .05, "Pneumonia": .04, "Malaria": .02}
FEVER = {"Malaria": 38.9, "Typhoid Fever": 39.0, "Pneumonia": 38.6, "Meningitis": 39.2, "Acute Respiratory Infection": 38.0,
         "Tuberculosis": 37.8, "Urinary Tract Infection": 37.9, "Cholera": 37.6, "Hepatitis B": 37.4, "HIV/AIDS": 37.5,
         "Diarrhoeal Disease": 37.8, "Sickle Cell Disease": 37.7}
FTYPES = ["Teaching Hospital", "Regional Hospital", "District Hospital", "Polyclinic", "Health Centre", "CHPS Compound"]
FCOST = {"Teaching Hospital": 2.5, "Regional Hospital": 1.6, "District Hospital": 1.0, "Polyclinic": .8, "Health Centre": .5, "CHPS Compound": .3}
SEVERE = {"Stroke", "Meningitis", "Road Traffic Injury", "Pregnancy Complications", "Cholera"}

# --------------------------------------------------------------------------- the clean "true" world
region_names = list(REGIONS)
rw = np.array([REGIONS[r][0] for r in region_names]); rw = rw / rw.sum()
region = rng.choice(region_names, N, p=rw)

days = pd.date_range("2023-01-01", "2024-12-31")
mw = {1: .95, 2: .9, 3: .95, 4: 1, 5: 1.05, 6: 1.15, 7: 1.2, 8: 1.15, 9: 1.1, 10: 1.1, 11: .95, 12: .95}
w = np.array([mw[d.month] for d in days]); w /= w.sum()
adm = days[rng.choice(len(days), N, p=w)]

grp = rng.choice(["u5", "5_17", "18_39", "40_59", "60p"], N, p=[.20, .10, .30, .22, .18])
age = np.empty(N, dtype=int)
for g, (lo, hi) in {"u5": (0, 4), "5_17": (5, 17), "18_39": (18, 39), "40_59": (40, 59)}.items():
    m = grp == g; age[m] = rng.integers(lo, hi + 1, m.sum())
m = grp == "60p"; age[m] = np.clip(60 + rng.gamma(2.2, 6, m.sum()).astype(int), 60, 98)
gender = rng.choice(["Male", "Female"], N, p=[.47, .53])

diag = np.empty(N, dtype=object)
for k in range(N):
    wts = {d: AGE_W[grp[k]].get(d, 0.0) for d in DISEASES}
    if gender[k] == "Female" and 15 <= age[k] <= 45:
        wts["Pregnancy Complications"] = 14 if grp[k] == "18_39" else 3
    r = region[k]
    if r in NORTH: wts["Meningitis"] *= 6; wts["Malaria"] *= 1.3; wts["Malnutrition"] *= 1.8
    if r in COAST: wts["Cholera"] *= 2.5
    if r in URBAN: wts["Hypertension"] *= 1.2; wts["Diabetes Mellitus"] *= 1.2; wts["Road Traffic Injury"] *= 1.5; wts["Typhoid Fever"] *= 1.3
    if 6 <= adm[k].month <= 10: wts["Malaria"] *= 1.6
    p = np.array([wts[d] for d in DISEASES], dtype=float); p /= p.sum()
    diag[k] = DISEASES[rng.choice(len(DISEASES), p=p)]
issues["pregnancy_male_rows"] = int(((diag == "Pregnancy Complications") & (gender == "Male")).sum())

ftype = np.empty(N, dtype=object)
for k in range(N):
    base = np.array([.08 if region[k] in TEACHING_REGIONS else 0, .18, .30, .14, .20, .10])
    if diag[k] in SEVERE: base *= np.array([2.5, 2, 1, .6, .2, .05])
    ftype[k] = FTYPES[rng.choice(6, p=base / base.sum())]

town = np.array([rng.choice(REGIONS[r][1]) for r in region])
facility = np.where(ftype == "Regional Hospital", [REGIONS[r][1][0] for r in region], town)
facility_name = np.array([f"{t} {ft}" for t, ft in zip(facility, ftype)], dtype=object)

atype = np.empty(N, dtype=object)
for k in range(N):
    if diag[k] in SEVERE: p = [.70, .10, .12, .08]
    elif diag[k] in ("Hypertension", "Diabetes Mellitus", "Asthma"): p = [.15, .60, .10, .15]
    else: p = [.35, .40, .12, .13]
    atype[k] = ["Emergency", "Outpatient", "Referral", "Inpatient"][rng.choice(4, p=p)]
los = np.array([0 if atype[k] == "Outpatient" else max(1, int(round(rng.lognormal(np.log(LOS_MEAN[diag[k]]), .45))))
                for k in range(N)])
disch = adm + pd.to_timedelta(los, unit="D")

# outcome
outcome = np.empty(N, dtype=object)
for k in range(N):
    dead = MORTALITY.get(diag[k], .02)
    ref = {"CHPS Compound": .25, "Health Centre": .18, "Polyclinic": .12}.get(ftype[k], .05)
    lama = .03
    rest = max(0, 1 - dead - ref - lama)
    outcome[k] = rng.choice(["Recovered", "Improved", "Referred", "Deceased", "Left Against Advice"],
                            p=[rest * .65, rest * .35, ref, dead, lama])

# vitals
temp = np.array([rng.normal(FEVER.get(d, 36.8), .6 if d in FEVER else .4) for d in diag]).clip(35, 41.5)
sys_bp = np.where(age < 12, rng.normal(95 + age * 1.2, 8), rng.normal(115 + .45 * (age - 30), 12))
sys_bp = sys_bp + np.select([diag == "Hypertension", diag == "Stroke", diag == "Diabetes Mellitus"], [35, 45, 12], 0)
sys_bp = np.clip(np.round(sys_bp), 60, 230)
dia_bp = np.clip(np.round(.62 * sys_bp + rng.normal(0, 6, N)), 35, 140)
hr = np.where(age < 10, rng.normal(120 - age * 3, 12), rng.normal(78, 10)) + np.where(np.isin(diag, list(FEVER)), 8, 0)
hr = np.clip(np.round(hr), 45, 190)
height = np.where(age < 2, 50 + age * 25 + rng.normal(0, 3, N),
         np.where(age < 18, 85 + (age - 2) * 7.5 + rng.normal(0, 6, N),
                  rng.normal(np.where(gender == "Male", 171, 160), 7, N)))
bmi = np.where(age < 18, rng.normal(16.5, 1.8, N), rng.normal(24.5, 4.2, N))
bmi = np.where(np.isin(diag, ["Malnutrition"]), rng.normal(13.5, 1.2, N), bmi).clip(11, 45)
weight = np.round(bmi * (height / 100) ** 2, 1)
height = np.round(height, 0)

cost = np.array([COST_BASE[d] * FCOST[f] * (1 + .25 * l) * rng.lognormal(0, .25)
                 for d, f, l in zip(diag, ftype, los)])
cost = np.round(cost, 2)

ins_p = np.where(atype == "Emergency", .0, 0.0)  # (placeholder, probabilities set per row below)
insurance = np.array([rng.choice(["NHIS", "Private Insurance", "Cash", "Uninsured"],
                                 p=[.62, .08, .18, .12] if age[k] >= 18 else [.70, .05, .12, .13]) for k in range(N)], dtype=object)

pid = np.array([f"PT-{i + 1:06d}" for i in range(N)], dtype=object)

# --------------------------------------------------------------------------- make it messy
def pick(frac, size=None):
    return rng.choice(N, size if size is not None else int(N * frac), replace=False)

# dates as text
od = np.array(adm.strftime("%Y-%m-%d"), dtype=object)
dd = np.array(disch.strftime("%Y-%m-%d"), dtype=object)
i = pick(.025); od[i] = adm[i].strftime("%d/%m/%Y"); issues["admission_date_ddmmyyyy"] = len(i)
i = pick(.010); od[i] = adm[i].strftime("%d-%b-%Y"); issues["admission_date_dd-Mon-yyyy"] = len(i)
i = pick(.003); od[i] = rng.choice(["unknown", "2024-13-45", "TBD", "31/02/2024"], len(i)); issues["admission_date_invalid"] = len(i)
i = pick(.025); dd[i] = disch[i].strftime("%d/%m/%Y"); issues["discharge_date_ddmmyyyy"] = len(i)
i = pick(.004); dd[i] = (adm[i] - pd.to_timedelta(rng.integers(1, 8, len(i)), unit="D")).strftime("%Y-%m-%d"); issues["discharge_before_admission"] = len(i)
i = pick(.03); dd[i] = None; issues["discharge_date_missing_still_admitted"] = len(i)

# impossible values and extreme outliers
age = age.astype(float)
i = pick(.003); age[i] = rng.choice([-3, 0, 150, 200, 999], len(i)); issues["age_impossible"] = len(i)
los = los.astype(float)
i = pick(.002); los[i] = rng.integers(150, 600, len(i)); issues["los_extreme_150_600_days"] = len(i)
i = pick(.003); los[i] = -rng.integers(1, 6, len(i)); issues["los_negative"] = len(i)
i = pick(.010); los[i] = np.round(los[i] + rng.integers(2, 6, len(i))); issues["los_mismatch_with_dates"] = len(i)
i = pick(.006); temp[i] = np.round(temp[i] * 9 / 5 + 32, 1); issues["temperature_in_fahrenheit"] = len(i)
i = pick(.002); temp[i] = rng.choice([0, 3.7, 420, 99999], len(i)); issues["temperature_impossible"] = len(i)
i = pick(.003); sys_bp[i] = rng.choice([0, 22, 2200, 400], len(i)); issues["systolic_impossible"] = len(i)
i = pick(.003); dia_bp[i] = rng.choice([0, 5, 1200, 300], len(i)); issues["diastolic_impossible"] = len(i)
i = pick(.012); k = i[sys_bp[i] > 0]; dia_bp[k] = sys_bp[k] + rng.integers(5, 40, len(k)); issues["diastolic_greater_than_systolic"] = len(k)
i = pick(.003); hr[i] = rng.choice([0, 12, 400, 999], len(i)); issues["heart_rate_impossible"] = len(i)
i = pick(.005); weight[i] = np.round(weight[i] * 2.2, 1); issues["weight_in_pounds"] = len(i)
i = pick(.003); weight[i] = rng.choice([-5, 0, 999, 5400], len(i)); issues["weight_impossible"] = len(i)
i = pick(.008); height[i] = np.round(height[i] / 100, 2); issues["height_in_metres"] = len(i)
i = pick(.003); height[i] = rng.choice([17, 1700, 0, 9999], len(i)); issues["height_impossible"] = len(i)
i = pick(.003); cost[i] = np.round(cost[i] * rng.integers(50, 300, len(i)), 2); issues["cost_extreme_high"] = len(i)
i = pick(.002); cost[i] = rng.choice([-500, 0, -75.5], len(i)); issues["cost_negative_or_zero"] = len(i)

# typos and inconsistent labels
def typos(arr, variants, frac, key):
    arr = arr.astype(object); n_changed = 0
    for k in pick(frac):
        v = arr[k]
        if v in variants:
            arr[k] = str(rng.choice(variants[v])); n_changed += 1
    issues[key] = n_changed
    return arr

diag = typos(diag, {
    "Malaria": ["malaria", "Malria", "MALARIA", "Malaria ", "Malarai"],
    "Typhoid Fever": ["Typhoid Feever", "Tyfoid", "typhoid", "Typhoid fever", "Thyphoid Fever"],
    "Hypertension": ["Hypertention", "Hypertenshion", "HTN", "hypertension", "High Blood Pressure"],
    "Diabetes Mellitus": ["Diabetis", "Diabetes Melitus", "DM", "diabetes", "Diabetes Mellitus "],
    "Pneumonia": ["Pnuemonia", "pneumonia", "Pneumonia ", "Pneumonai"],
    "Tuberculosis": ["TB", "Tuberclosis", "tuberculosis", "Tubercolosis"],
    "HIV/AIDS": ["HIV AIDS", "Hiv/Aids", "HIV", "HIV/ AIDS"],
    "Cholera": ["Colera", "Cholera ", "cholera", "Choleraa"],
    "Sickle Cell Disease": ["Sickle cell", "Sickel Cell Disease", "SCD", "sickle cell disease"],
    "Anaemia": ["Anemia", "Aneamia", "Anaemia ", "anaemia"],
    "Urinary Tract Infection": ["UTI", "Urinary Tract Infction", "urinary tract infection", "Urinay Tract Infection"],
    "Diarrhoeal Disease": ["Diarrhea", "Diarhoea", "Diarrheal Disease", "diarrhoeal disease"],
    "Acute Respiratory Infection": ["ARI", "Acute Resp. Infection", "Acute Respiratory Infection ", "acute respiratory infection"],
    "Meningitis": ["Menengitis", "Meningitus", "meningitis", "Meningitis "],
    "Stroke": ["Stoke", "CVA", "stroke", "Strokee"],
    "Asthma": ["Asthama", "Asma", "asthma", "Asthma "],
    "Hepatitis B": ["Hepatits B", "Hep B", "hepatitis b", "Hepatitis  B"],
    "Malnutrition": ["Malnutriton", "malnutrition", "Malnutrition ", "Mal-nutrition"],
    "Road Traffic Injury": ["RTA", "Road Traffic Accident", "road traffic injury", "Road Traffic Injuries"],
    "Pregnancy Complications": ["Pregnancy Complication", "Preg. Complications", "pregnancy complications", "Pregnancy Complicatons"],
}, .24, "diagnosis_misspelled_or_inconsistent")

region = typos(region, {
    "Greater Accra": ["Greater accra", "G. Accra", "Gt. Accra", "Greater Acra", "GREATER ACCRA"],
    "Ashanti": ["Ashante", "Asante", "ASHANTI", "ashanti "], "Western North": ["Western North ", "W/North", "Western-North", "Western Nort"],
    "Upper East": ["Upper-East", "U/East", "upper east", "Upper Eats"], "Upper West": ["U/West", "Upper west", "Upper-West", "Uper West"],
    "Volta": ["Volta ", "Voltta", "volta"], "Northern": ["Nothern", "Northern ", "northern"], "Central": ["Centrel", "central", "Central "],
    "Eastern": ["Eastren", "eastern", "Eastern "], "Western": ["Westren", "western", "Western "],
    "Savannah": ["Savanna", "Savannah ", "savannah"], "North East": ["North-East", "N/East", "Northeast", "North east"],
    "Bono East": ["Bono-East", "Brong Ahafo", "bono east"], "Bono": ["Bono ", "Brong-Ahafo", "bono"], "Ahafo": ["Ahafo ", "Brong Ahafo", "ahafo"],
    "Oti": ["oti", "Oti "],
}, .18, "region_misspelled_or_legacy")

gender = typos(gender, {"Male": ["M", "male", "MALE", "Male ", "Mlae"], "Female": ["F", "female", "Femle", "FEMALE", "Female "]},
               .15, "gender_inconsistent")
ftype = typos(ftype, {"Polyclinic": ["Poly Clinic", "polyclinic", "Polyclinic "], "Health Centre": ["Health Center", "Health centre", "health centre"],
                      "CHPS Compound": ["Chps", "CHPS compound", "CHPS"], "District Hospital": ["District hospital", "District Hosp."],
                      "Regional Hospital": ["Regional hospital", "Regional Hosp."], "Teaching Hospital": ["Teaching hospital", "Teaching Hosp."]},
              .12, "facility_type_inconsistent")
facility_name = typos(facility_name, {n: [n.lower(), n + " ", n.upper()] for n in set(facility_name)}, .06, "facility_name_case_or_spaces")
atype = typos(atype, {"Emergency": ["Emergncy", "emergency", "EMERGENCY", "Emergency "], "Outpatient": ["OPD", "Out-patient", "outpatient", "Outpatient "],
                      "Referral": ["referral", "Refferal", "Referral "], "Inpatient": ["In-patient", "inpatient", "Inpatient "]},
              .12, "admission_type_inconsistent")
insurance = typos(insurance, {"NHIS": ["nhis", "N.H.I.S", "NHIS ", "Nhis"], "Private Insurance": ["Private", "private insurance"],
                              "Cash": ["Self-pay", "cash", "Self Pay"], "Uninsured": ["No Insurance", "uninsured", "Uninsured "]},
                  .15, "insurance_type_inconsistent")
outcome = typos(outcome, {"Recovered": ["Recoverd", "recovered", "Recovered "], "Improved": ["Improoved", "improved", "Improved "],
                          "Referred": ["Reffered", "referred", "Referred "], "Deceased": ["Dead", "Died", "deceased", "Deceased "],
                          "Left Against Advice": ["LAMA", "Left against advice", "Left Against Advise"]},
                .15, "outcome_inconsistent")

# missing values, with several different "null" styles, and some patterns (not purely random)
NUM_TOK = (["", "N/A", "null", "None", "-", "NaN"], [.50, .15, .10, .10, .08, .07])
TXT_TOK = (["", "None", "null", "Not recorded", "unknown", "N/A"], [.40, .15, .10, .15, .10, .10])

def to_obj(a): return np.array(a, dtype=object)
cols = {
    "patient_id": pid, "admission_date": od, "discharge_date": dd, "age": to_obj(age), "gender": gender, "region": region,
    "facility_name": facility_name, "facility_type": ftype, "diagnosis": diag, "admission_type": atype,
    "length_of_stay_days": to_obj(los), "temperature_c": to_obj(np.round(temp, 1)), "systolic_bp": to_obj(sys_bp),
    "diastolic_bp": to_obj(dia_bp), "heart_rate": to_obj(hr), "weight_kg": to_obj(weight), "height_cm": to_obj(height),
    "insurance_type": insurance, "treatment_cost_ghs": to_obj(cost), "outcome": outcome,
}
low_level = np.isin(np.array(typos(np.array(ftype, dtype=object), {}, 0, "_tmp"), dtype=object).astype(str),
                    ["CHPS Compound", "Chps", "CHPS compound", "CHPS", "Health Centre", "Health Center", "Health centre", "health centre"])
miss_p = {  # base probability of a missing value per column
    "age": .03, "gender": .015, "region": .01, "facility_name": .01, "diagnosis": .02, "admission_type": .02,
    "length_of_stay_days": .04, "temperature_c": .05, "systolic_bp": .06, "diastolic_bp": .06, "heart_rate": .05,
    "weight_kg": .07, "height_cm": .09, "insurance_type": .12, "treatment_cost_ghs": .04, "outcome": .02, "admission_date": .003,
}
text_cols = {"gender", "region", "facility_name", "diagnosis", "admission_type", "insurance_type", "outcome", "admission_date"}
for c, p0 in miss_p.items():
    p = np.full(N, p0)
    if c in ("weight_kg", "height_cm", "temperature_c", "systolic_bp", "diastolic_bp", "heart_rate"):
        p = np.where(low_level, p0 * 2.2, p0)                       # vitals missing more at small facilities
    if c in ("weight_kg", "height_cm"):
        p = np.where(atype == "Outpatient", p * 1.6, p)
    if c == "insurance_type":
        p = np.where(atype == "Emergency", p0 * 2, p0)
    hit = np.where(rng.random(N) < np.clip(p, 0, .6))[0]
    toks, tp = TXT_TOK if c in text_cols else NUM_TOK
    cols[c][hit] = rng.choice(toks, len(hit), p=tp)
    issues[f"missing_{c}"] = int(len(hit))

df = pd.DataFrame(cols)
df.loc[:, :] = df.where(df.notna(), None)

# --------------------------------------------------------------------------- duplicates
exact = df.sample(N_EXACT, random_state=SEED)
key_src = df.sample(N_KEY, random_state=SEED + 1)
key_a = key_src.iloc[: N_KEY // 2].copy()            # billing correction: same patient_id, different cost
key_a["treatment_cost_ghs"] = [round(float(x) * 1.1, 2) if str(x).replace(".", "", 1).replace("-", "", 1).isdigit() else x
                               for x in key_a["treatment_cost_ghs"]]
key_b = key_src.iloc[N_KEY // 2:].copy()             # less complete copy: vitals dropped
key_b[["temperature_c", "systolic_bp", "diastolic_bp", "heart_rate"]] = ""
out = pd.concat([df, exact, key_a, key_b], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
issues["exact_duplicate_rows_added"] = N_EXACT
issues["key_duplicate_patient_ids_added"] = N_KEY
assert out.shape == (N_TOTAL, 20), out.shape

out.to_csv("ghana_hospital_data.csv", index=False, encoding="utf-8")
json.dump({k: int(v) for k, v in issues.items()}, open("_issues.json", "w"), indent=1)
print("Saved ghana_hospital_data.csv", out.shape)
