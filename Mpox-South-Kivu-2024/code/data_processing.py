#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import re
import unicodedata

from typing import Optional, Union

# -------------------------------------------------------------------------------------
# CHANGE THIS

FILEPATH = "./dataset.csv"  # to the raw data file itself
OUTPUT_FILEPATH = '.'  # folder where clean data and dictionary will be saved

# -------------------------------------------------------------------------------------
# String extraction from free text columns

locations = {
    "face": ["face"],
    "torso": ["torse", "torso"],
    "limbs": ["limbs"],
    "genital": [
        "genital",
        "genitalarea",
        r"esion\s+genitale",
        r"genetal\s+area",
        r"geniatal\s+area",
        r"genital\s+area",
        r"genital\s+areao",
        r"genitale\s+area",
        r"ginital\s+area",
        r"lesion\s+genital",
        r"lesion\s+genitale",
        r"lesion\s+genitaleo",
        r"oedeme\s+genital"
    ]
}

locations_alt = {
    "face": "|".join(locations["face"]),
    "torso": "|".join(locations["torso"]),
    "limbs": "|".join(locations["limbs"]),
    "genital": "|".join(locations["genital"]),
}

locations_all = "|".join(locations_alt.values())
loc_pattern = re.compile(rf"(?i)\b(?P<loc>{locations_all})\s*,?\s*(?P<num>\d+)\b")

outcomes_alt = {
    "resolved": r"lesions\s+are\s+completely\s+resolved",
    "eye": r"eyes|eye",
    "psychological": r"psychological|psychologiques",
    "gynobs": r"gynobs|gyneco-obstetrical",
    "died": r"died|decede",
    "readmitted": "readmises",
    "discharged": r"sortie|sorti|sotie|sorie|sotrie",
}

outcomes_pattern = "|".join(f"(?P<{k}>{v})" for k, v in outcomes_alt.items())
outcomes_pattern = re.compile(rf"(?i)\b({outcomes_pattern})\b")

# Date pattern
date_pattern = re.compile(r"(?P<date>\b\d{1,2}/\d{1,2}/\d{4})")


def normalize_text(s: str, strip_accents: bool = True):
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)  # collapse whitespace
    if strip_accents:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def extract(text: str, key: Optional[str] = None):
    raw = normalize_text(str(text), strip_accents=True)

    # Initalise counts with 0
    extracted = {"face": 0, "torso": 0, "limbs": 0, "genital": 0}

    # date
    m = date_pattern.search(raw)
    extracted["date"] = m.group("date") if m else None

    # locations: iterate finditer and map back to canonical key
    for m in loc_pattern.finditer(raw):
        found = m.group("loc").lower()
        num = int(m.group("num"))
        # determine canonical location by testing against each alt set
        if re.fullmatch(locations_alt["face"], found, flags=re.I):
            extracted["face"] = num
        elif re.fullmatch(locations_alt["torso"], found, flags=re.I):
            extracted["torso"] = num
        elif re.fullmatch(locations_alt["limbs"], found, flags=re.I):
            extracted["limbs"] = num
        elif re.fullmatch(locations_alt["genital"], found, flags=re.I):
            extracted["genital"] = num

    extracted["total_count"] = (
        extracted["face"]
        + extracted["torso"]
        + extracted["limbs"]
        + extracted["genital"]
    )
    extracted["severity"] = (
        "None" if extracted["total_count"] < 1
        else "Mild" if extracted["total_count"] < 25
        else "Moderate" if extracted["total_count"] < 100
        else "Severe" if extracted["total_count"] < 251
        else "Critical"
    )

    resolved = (
        True
        if any(m.group("resolved") for m in outcomes_pattern.finditer(raw))
        else False
    )
    outcomes = {
        k: (
            True
            if any(m.group(k) for m in outcomes_pattern.finditer(raw))
            else False
        )
        for k in [
            "resolved",
            "eye",
            "psychological",
            "gynobs",
            "died",
            "readmitted",
            "discharged"
        ]
    }

    extracted["resolved"] = resolved and not any(outcomes.values())
    extracted = {**extracted, **outcomes}

    if key:
        return extracted.get(key, None)
    return extracted


def fix_date_error(s: Union[str, pd.Series]):
    """Some months entered with 3 digits e.g. 01/010/2024. Assume the
    leading 0 in the month is an error"""
    if isinstance(s, str):
        return re.sub(
            r'\b(\d{1,2})/0(\d{2})/(\d{4})\b',
            r'\1/\2/\3',
            s
        )
    if isinstance(s, pd.Series):
        return s.str.replace(
            r'\b(\d{1,2})/0(\d{2})/(\d{4})\b',
            r'\1/\2/\3',
            regex=True
        )
    return


# -------------------------------------------------------------------------------------
# Read data

df = pd.read_csv(FILEPATH, dtype={"CODE": "str"})

# Get rid of the key at the bottom of the file
df = df.loc[df["CODE"].notna()]

# Subjids are not unique!
df.loc[df["CODE"].duplicated(keep="first"), "CODE"] += ".1"

days = list(range(1, 15)) + [21, 28, 60, 120, 180, 240, 300, 360]

# -------------------------------------------------------------------------------------
# Create the new dataframe

columns = [
    "subjid",
    "demog_sex",
    "demog_agegroup",
    "demog_age",
    "demog_country_iso",
    "dates_admdate",
    "dates_onset_reldate",
    "outco_outcome",
    "outco_reldate",
    "outco_discharged_reldate",  # hospital discharge
    "outco_lesion_resolution",
    "outco_lesion_resolution_reldate",   # no censored times
    "comor_hiv",
    "preg_pregnant",
    "preg_outcome",
    "expo14_type",
    "onset_firstsym",
    "compl_eye",
    "compl_eye_reldate",
    "compl_psychological",
    "compl_psychological_reldate",
    "compl_gynobs",
    "compl_gynobs_reldate",
    "compl_readmitted",
    "compl_readmitted_reldate",
    "compl_extendedhosp",
    "compl_extendedhosp_reldate",
    "compl_any",
    "compl_any_reldate"
]
for d in days:
    columns += [
        f"D{d}_lesion___face",
        f"D{d}_lesion___torso",
        f"D{d}_lesion___genital",
        f"D{d}_lesion___limbs",
        f"D{d}_severity",
        f"D{d}_lesioncount"
    ]

df_map = pd.DataFrame(columns=columns)
df_map["subjid"] = df["CODE"].copy()

# -------------------------------------------------------------------------------------
# Demographics
df_map["demog_sex"] = df["SEXE"].apply(lambda x: x.lower().strip()).map(
    {
        "masculin": "Male",
        "féminin": "Female"
    }
)

df.loc[(df["AGE EN MOIS"] == "15 Jours"), "AGE EN MOIS"] = 0.5
ind = df["AGE EN ANNEE"].isna()
df.loc[ind, "AGE EN ANNEE"] = (df.loc[ind, "AGE EN MOIS"].astype(float) / 12)

df_map["demog_age"] = df["AGE EN ANNEE"].copy()
ind = df["AGE EN ANNEE"].isna() & (df["AGE EN MOIS"] != "15 Jours")
df_map.loc[(df["AGE EN MOIS"] == "15 Jours"), "demog_age"] = 0.5 / 12
df_map.loc[ind, "demog_age"] = (df.loc[ind, "AGE EN MOIS"].astype(float) / 12)

df_map["demog_agegroup"] = df_map["demog_age"].apply(
    lambda x:
        "0-4" if x < 5
        else "5-17" if x < 18
        else np.nan if np.isnan(x)
        else "18-79"
)

df_map["demog_country_iso"] = "COD"

# -------------------------------------------------------------------------------------
# Dates

df_map["dates_admdate"] = df["DATE D'ADMISSION"].apply(
    pd.to_datetime, errors="coerce", dayfirst=True
)

df_map["dates_onset_reldate"] = (
    df["ONSET DATE"]
    .apply(fix_date_error)
    .apply(pd.to_datetime, errors="coerce", dayfirst=True)
    .sub(df_map["dates_admdate"])
)
# Onset should not be after admission, some data issues
df_map.loc[
    df_map["dates_onset_reldate"] > pd.Timedelta(0),
    "dates_onset_reldate"
] = np.nan

# This is used later
daily_dates = (
    df[[f"J{d}" for d in days]]
    .apply(lambda x: x.apply(extract, key="date"), axis=0)
    .apply(fix_date_error)
    .apply(pd.to_datetime, errors="coerce", dayfirst=True)
)

# Should be increasing for each patient and after admission date
nan_mask = daily_dates.diff(axis=1) < pd.Timedelta(0)
nan_mask |= daily_dates.sub(df_map["dates_admdate"], axis=0) < pd.Timedelta(0)
daily_dates[nan_mask] = np.nan

# Fill missing with admission date + (x - 1) for column J{x}
replacement_dates = (
    pd.DataFrame(
        [[pd.Timedelta(d - 1, unit="day") for d in days] for x in df.index],
        index=df.index,
        columns=[f"J{d}" for d in days]
    )
    .add(df_map["dates_admdate"], axis=0)
)
daily_dates = daily_dates.fillna(replacement_dates)

# -------------------------------------------------------------------------------------
# Outcomes

df_map["outco_outcome"] = df["OUTCOME"].fillna("Still hospitalised")
df_map["outco_outcome"] = df_map["outco_outcome"].replace(
    {"Discharged  alive": "Discharged alive"}
)

df_map["outco_reldate"] = (
    df["MPOX OUTCOME DATE"]
    # .apply(lambda x: "/".join([y.lstrip("0") for y in x.split("/")]))  # bug fix
    .apply(fix_date_error)
    .apply(pd.to_datetime, errors="coerce", dayfirst=True)
    .sub(df_map["dates_admdate"])
)
# Outcomes should not be after admission (no errors currently)
df_map.loc[
    df_map["outco_reldate"] < pd.Timedelta(0),
    "outco_reldate"
] = np.nan

df_map["outco_lesion_resolution"] = df["MPOX PAT OUTCOME"].apply(extract, key="resolved")
df_map["outco_lesion_resolution_reldate"] = (
    df["MPOX PAT OUTCOME DATE"]
    .apply(fix_date_error)
    .apply(pd.to_datetime, errors="coerce", dayfirst=True)
    .sub(df_map["dates_admdate"])
    .where(df_map["outco_lesion_resolution"])
)
# Outcomes should not be after admission (no errors currently)
df_map.loc[
    df_map["outco_lesion_resolution_reldate"] < pd.Timedelta(0),
    "outco_lesion_resolution_reldate"
] = np.nan

mask = df[[f"J{d}" for d in days]].apply(
    lambda x: x.apply(extract, key="discharged"),
    axis=0
)
df_map["outco_discharged_reldate"] = df_map["outco_discharged_reldate"].astype('<m8[ns]')
df_map.loc[
    mask.any(axis=1),
    "outco_discharged_reldate"
] = (
    daily_dates
    .where(mask)
    .min(axis=1)
    .sub(df_map["dates_admdate"])
    .loc[mask.any(axis=1)]
)
# Outcomes should not be after admission (no errors currently)
df_map.loc[
    df_map["outco_discharged_reldate"] < pd.Timedelta(0),
    "outco_discharged_reldate"
] = np.nan

# -------------------------------------------------------------------------------------
# Comorbidities / pregnancy / transmission / onset of symptoms

df_map["comor_hiv"] = df["VIH"].map({"Positif": True, "Négatif": False})

df_map["preg_pregnant"] = df["PREGNANCY STATUS"].map(
    {"NO": False, "YES": True}
).astype(object)

# Insert NaN for Sex = Male / Age group = 0-4
df_map.loc[
    df_map["demog_sex"].isin(["Male"])
    | df_map["demog_agegroup"].isin(["0-4"]),
    "preg_pregnant"
] = np.nan

df_map["preg_outcome"] = df["PREGNANCY OUTCOME"].map(
    {
        "Childbirth,  and  baby born with pox and  cured": "Live birth",
        "Abortion,  3rd  trimester": "Termination",
        "Abortion, 2nd  trimester ": "Termination",
        "Abortion, 1st  trimester ": "Termination",
        "Childbirth,  but newborn died": "Termination",
        "Childbirth,  but  newborn died": "Neonatal death",
        "Childbirth, and  newborn  alive": "Live birth",
        "Childbirth,  but  newborn alive": "Live birth",
        "Abortion, 2nd  trimester  ": "Termination",
        "Abortion,  2rd  trimester": "Termination",
        "NO ": np.nan,
        "NO": np.nan,
        "Childbirth, but  newborn died": "Neonatal death",
        "Abortion,  1st  trimester": "Termination",
    }
)
df_map.loc[df_map["preg_outcome"].notna(), "preg_pregnant"] = True

# Transmission
df_map["expo14_type"] = (
    df["CONTACT"]
    .apply(lambda x: x.lower().strip())
    .map(
        {
            "sexuel": "Sexual transmission",
            "communautaire": "Community contact",
            "vertical": "Vertical transmission",
            "animal": "Unknown / other",
            "inconnu": "Unknown / other",
        }
    )
    .fillna("Unknown / other")
)

df_map["onset_firstsym"] = df["FIRST SYMPTOM"].map(
    {"Rush": "Rash", "Fever": "Fever"}
)

# -------------------------------------------------------------------------------------
# Complications

df_map["compl_eye"] = df["MPOX PAT OUTCOME"].apply(extract, key="eye")
df_map["compl_eye_reldate"] = (
    df[["MPOX PAT OUTCOME", "MPOX PAT OUTCOME DATE"]]
    .apply(
        lambda x:
            x["MPOX PAT OUTCOME DATE"]
            if extract(x["MPOX PAT OUTCOME"], key="eye")
            else np.nan,
        axis=1,
    )
    .apply(fix_date_error)
    .apply(pd.to_datetime, errors="coerce", dayfirst=True)
    .astype('<M8[ns]')
    .sub(df_map["dates_admdate"])
)

mask = df[[f"J{d}" for d in days]].apply(
    lambda x: x.apply(extract, key="eye"),
    axis=0
)
df_map.loc[mask.any(axis=1), "compl_eye"] = True
df_map.loc[
    mask.any(axis=1),
    "compl_eye_reldate"
] = pd.concat(
    [
        (
            daily_dates
            .where(mask)
            .min(axis=1)
            .sub(df_map["dates_admdate"])
            .loc[mask.any(axis=1)]
        ),
        df_map.loc[mask.any(axis=1), "compl_eye_reldate"],
    ],
    axis=1
).min(axis=1)

df_map["compl_psychological"] = df["MPOX PAT OUTCOME"].apply(extract, key="psychological")
df_map["compl_psychological_reldate"] = (
    df[["MPOX PAT OUTCOME", "MPOX PAT OUTCOME DATE"]]
    .apply(
        lambda x:
            x["MPOX PAT OUTCOME DATE"]
            if extract(x["MPOX PAT OUTCOME"], key="psychological")
            else np.nan,
        axis=1,
    )
    .apply(fix_date_error)
    .apply(pd.to_datetime, errors="coerce", dayfirst=True)
    .astype('<M8[ns]')
    .sub(df_map["dates_admdate"])
)

mask = df[[f"J{d}" for d in days]].apply(
    lambda x: x.apply(extract, key="eye"),
    axis=0
)
df_map.loc[mask.any(axis=1), "compl_psychological"] = True
df_map.loc[
    mask.any(axis=1),
    "compl_psychological_reldate"
] = pd.concat(
    [
        (
            daily_dates
            .where(mask)
            .min(axis=1)
            .sub(df_map["dates_admdate"])
            .loc[mask.any(axis=1)]
        ),
        df_map.loc[mask.any(axis=1), "compl_psychological_reldate"],
    ],
    axis=1
).min(axis=1)

df_map["compl_gynobs"] = df["MPOX PAT OUTCOME"].apply(extract, key="gynobs")
df_map["compl_gynobs_reldate"] = (
    df[["MPOX PAT OUTCOME", "MPOX PAT OUTCOME DATE"]]
    .apply(
        lambda x:
            x["MPOX PAT OUTCOME DATE"]
            if extract(x["MPOX PAT OUTCOME"], key="gynobs")
            else np.nan,
        axis=1,
    )
    .apply(fix_date_error)
    .apply(pd.to_datetime, errors="coerce", dayfirst=True)
    .astype('<M8[ns]')
    .sub(df_map["dates_admdate"])
)

mask = df[[f"J{d}" for d in days]].apply(
    lambda x: x.apply(extract, key="gynobs"),
    axis=0
)
df_map.loc[mask.any(axis=1), "compl_gynobs"] = True
df_map.loc[
    mask.any(axis=1),
    "compl_gynobs_reldate"
] = pd.concat(
    [
        (
            daily_dates
            .where(mask)
            .min(axis=1)
            .sub(df_map["dates_admdate"])
            .loc[mask.any(axis=1)]
        ),
        df_map.loc[mask.any(axis=1), "compl_gynobs_reldate"],
    ],
    axis=1
).min(axis=1)

# Insert NaN for Sex = Male / Age group = 0-4
df_map["compl_gynobs"] = df_map["compl_gynobs"].astype(object)
df_map.loc[
    df_map["demog_sex"].isin(["Male"])
    | df_map["demog_agegroup"].isin(["0-4"]),
    ["compl_gynobs", "compl_gynobs_reldate"]
] = np.nan

df_map["compl_readmitted"] = df["MPOX PAT OUTCOME"].apply(extract, key="readmitted")
df_map["compl_readmitted_reldate"] = (
    df[["MPOX PAT OUTCOME", "MPOX PAT OUTCOME DATE"]]
    .apply(
        lambda x:
            x["MPOX PAT OUTCOME DATE"]
            if extract(x["MPOX PAT OUTCOME"], key="readmitted")
            else np.nan,
        axis=1,
    )
    .apply(fix_date_error)
    .apply(pd.to_datetime, errors="coerce", dayfirst=True)
    .astype('<M8[ns]')
    .sub(df_map["dates_admdate"])
)

mask = df[[f"J{d}" for d in days]].apply(
    lambda x: x.apply(extract, key="readmitted"),
    axis=0
)
df_map.loc[mask.any(axis=1), "compl_readmitted"] = True
df_map.loc[
    mask.any(axis=1),
    "compl_readmitted_reldate"
] = pd.concat(
    [
        (
            daily_dates
            .where(mask)
            .min(axis=1)
            .sub(df_map["dates_admdate"])
            .loc[mask.any(axis=1)]
        ),
        df_map.loc[mask.any(axis=1), "compl_readmitted_reldate"],
    ],
    axis=1
).min(axis=1)


df_map["compl_extendedhosp"] = df_map["outco_discharged_reldate"] > pd.Timedelta(14, unit="days")
df_map["compl_extendedhosp_reldate"] = (
    pd.Series(pd.Timedelta(14, unit="days"), index=df_map.index)
    .where(df_map["compl_extendedhosp"])
)

# Any complication
df_map["compl_any"] = pd.concat(
    [
        df_map["outco_outcome"] == "Died",
        df_map["compl_eye"],
        df_map["compl_psychological"],
        df_map["compl_gynobs"],
        df_map["compl_readmitted"],
        df_map["compl_extendedhosp"]
    ],
    axis=1
).any(axis=1)
df_map["compl_any_reldate"] = pd.concat(
    [
        df_map["outco_reldate"].where(df_map["outco_outcome"] == "Died"),
        df_map["compl_eye_reldate"],
        df_map["compl_psychological_reldate"],
        df_map["compl_gynobs_reldate"],
        df_map["compl_readmitted_reldate"],
        df_map["compl_extendedhosp_reldate"]
    ],
    axis=1
).min(axis=1)

# Moved this to analysis - cause-specific Cox
# # -------------------------------------------------------------------------------------
# # Update lesion resolution with complications / SAE
# # Can only use information from the past (i.e. if complications appear in daily
# # data before the lesion resolution outcome is recorded)
#
# df_map.loc[
#     df_map["outco_time_to_event"] > df_map["compl_any_reldate"],
#     "outco_lesion_resolution"
# ] = False
# df_map.loc[
#     df_map["outco_time_to_event"] > df_map["compl_any_reldate"],
#     "outco_time_to_event"
# ] = df_map.loc[
#     df_map["outco_time_to_event"] > df_map["compl_any_reldate"],
#     "compl_any_reldate"
# ]

# -------------------------------------------------------------------------------------
# Lesion locations and severity

# J1 is the day of admission (i.e. same day as baseline)
for location in ["face", "torso", "genital", "limbs"]:
    df_map[f"D1_lesion___{location}"] = (
        df["BASELINE OF LESIONS SITES  AND COUNT"].apply(extract, key=location)
        | df["J1"].apply(extract, key=location)
    )

df_map["D1_lesioncount"] = (
    df["BASELINE OF LESIONS SITES  AND COUNT"].apply(extract, key="total_count")
    | df["J1"].apply(extract, key="total_count")
)
df_map["D1_severity"] = df_map["D1_lesioncount"].apply(
    lambda x:
        "None" if x < 1
        else "Mild" if x < 25
        else "Moderate" if x < 100
        else "Severe" if x < 251
        else "Critical"
)
for location in ["face", "torso", "genital", "limbs"]:
    df_map[f"D1_lesion___{location}"] = df_map[f"D1_lesion___{location}"] > 0

for d in days:
    if d != 1:  # involves baseline, as above
        df_map[f"D{d}_lesioncount"] = df[f"J{d}"].apply(extract, key="total_count")
        df_map[f"D{d}_severity"] = df[f"J{d}"].apply(extract, key="severity")
        for location in ["face", "torso", "genital", "limbs"]:
            df_map[f"D{d}_lesion___{location}"] = df[f"J{d}"].apply(extract, key=location) > 0

# -------------------------------------------------------------------------------------
# Retain only patients admitted before 1st Nov 2024 as per paper
max_date = pd.to_datetime("2024-11-01")
include_ind = (
    df["ENROLMENT DATE"]
    .apply(fix_date_error)
    .apply(pd.to_datetime, errors="coerce", dayfirst=True)
) < max_date
df_map = df_map.loc[include_ind]

# -------------------------------------------------------------------------------------
# Change _reldate columns to integer

reldate_columns = [c for c in df_map.columns if c.endswith('_reldate')]
for column in reldate_columns:
    df_map[column] = df_map[column].dt.days

# -------------------------------------------------------------------------------------
# Save data

df_map.to_csv(
    os.path.join(OUTPUT_FILEPATH, "df_map.csv"), index=False)

# -------------------------------------------------------------------------------------
# Dictionary

dictionary = [
    {
        "field_name": "subjid",
        "field_type": "freetext",
        "field_label": "Participant Identification Number (PIN)",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None,
    },
    {
        "field_name": "demog_sex",
        "field_type": "categorical",
        "field_label": "Sex at birth",
        "answer_options": "Male, Female",
        "parent": None,
        "branching_logic": None,
        "section": None,
    },
    {
        "field_name": "demog_sex___male",
        "field_type": "binary",
        "field_label": "Male",
        "answer_options": None,
        "parent": "demog_sex",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "demog_sex___female",
        "field_type": "binary",
        "field_label": "Female",
        "answer_options": None,
        "parent": "demog_sex",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "demog_agegroup",
        "field_type": "categorical",
        "field_label": "Age in years",
        "answer_options": "0-4, 5-17, 18-79",
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "demog_agegroup___0_4",
        "field_type": "binary",
        "field_label": "0-4",
        "answer_options": None,
        "parent": "demog_agegroup",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "demog_agegroup___5_17",
        "field_type": "binary",
        "field_label": "5-17",
        "answer_options": None,
        "parent": "demog_agegroup",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "demog_agegroup___18_79",
        "field_type": "binary",
        "field_label": "18-79",
        "answer_options": None,
        "parent": "demog_agegroup",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "demog_country_iso",
        "field_type": "categorical",
        "field_label": "Country ISO",
        "answer_options": "COD",
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "demog_country_iso___cod",
        "field_type": "binary",
        "field_label": "Congo, Dem. Rep.",
        "answer_options": None,
        "parent": "demog_country_iso",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "dates_admdate",
        "field_type": "date",
        "field_label": "Date of admission/observation",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "dates_onset_reldate",
        "field_type": "number",
        "field_label": "Days from admission to onset of symptoms",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "outco_outcome",
        "field_type": "categorical",
        "field_label": "Outcome",
        "answer_options": "Discharged alive, Died",
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "outco_outcome___discharged_alive",
        "field_type": "binary",
        "field_label": "Discharged alive",
        "answer_options": None,
        "parent": "outco_outcome",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "outco_outcome___died",
        "field_type": "binary",
        "field_label": "Died",
        "answer_options": None,
        "parent": "outco_outcome",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "outco_reldate",
        "field_type": "number",
        "field_label": "Days from admission to outcome",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "outco_discharged_reldate",
        "field_type": "number",
        "field_label": "Days from admission to hospital discharge",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "outco_lesion_resolution",
        "field_type": "binary",
        "field_label": "Lesion resolution without serious complications",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "outco_lesion_resolution_reldate",
        "field_type": "number",
        "field_label": "Number of days from admission to lesion resolution without serious complications",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "comor_hiv",
        "field_type": "binary",
        "field_label": "HIV status positive",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "preg_pregnant",
        "field_type": "binary",
        "field_label": "Pregnancy",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "preg_outcome",
        "field_type": "categorical",
        "field_label": "Pregnancy outcome",
        "answer_options": "Live birth, Termination, Neonatal death",
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "preg_outcome___live_birth",
        "field_type": "binary",
        "field_label": "Live birth",
        "answer_options": None,
        "parent": "preg_outcome",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "preg_outcome___termination",
        "field_type": "binary",
        "field_label": "Termination",
        "answer_options": None,
        "parent": "preg_outcome",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "preg_outcome___neonatal_death",
        "field_type": "binary",
        "field_label": "Neonatal death",
        "answer_options": None,
        "parent": "preg_outcome",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "expo14_type",
        "field_type": "categorical",
        "field_label": "Transmission route",
        "answer_options": "Sexual transmission, Community contact, Vertical transmission, Other, Unknown",
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "expo14_type___sexual_transmission",
        "field_type": "binary",
        "field_label": "Sexual transmission",
        "answer_options": None,
        "parent": "expo14_type",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "expo14_type___community_contact",
        "field_type": "binary",
        "field_label": "Community contact",
        "answer_options": None,
        "parent": "expo14_type",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "expo14_type___vertical_transmission",
        "field_type": "binary",
        "field_label": "Vertical transmission",
        "answer_options": None,
        "parent": "expo14_type",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "expo14_type___unknown_other",
        "field_type": "binary",
        "field_label": "Unknown / other",
        "answer_options": None,
        "parent": "expo14_type",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "onset_firstsym",
        "field_type": "categorical",
        "field_label": "First symptom",
        "answer_options": "Fever, Rash",
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "onset_firstsym___fever",
        "field_type": "binary",
        "field_label": "Fever",
        "answer_options": None,
        "parent": "onset_firstsym",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "onset_firstsym___rash",
        "field_type": "binary",
        "field_label": "Rash",
        "answer_options": None,
        "parent": "onset_firstsym",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_eye",
        "field_type": "binary",
        "field_label": "Eye complications",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_eye_reldate",
        "field_type": "number",
        "field_label": "Number of days from admission to eye complications",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_psychological",
        "field_type": "binary",
        "field_label": "Psychological complications",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_psychological_reldate",
        "field_type": "number",
        "field_label": "Number of days from admission to psychological complications",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_gynobs",
        "field_type": "binary",
        "field_label": "Gyneco-obstetrical complications",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_gynobs_reldate",
        "field_type": "number",
        "field_label": "Number of days from admission to gyneco-obstetrical complications",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_readmitted",
        "field_type": "binary",
        "field_label": "Hospital readmission",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_readmitted_reldate",
        "field_type": "number",
        "field_label": "Number of days from admission to hospital readmission",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_extendedhosp",
        "field_type": "binary",
        "field_label": "Extended hospitalisation (>14 days)",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_extendedhosp_reldate",
        "field_type": "number",
        "field_label": "Number of days from admission to extended hospitalisation outcome",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_any",
        "field_type": "binary",
        "field_label":
            "Any complication (eye, pyschological, "
            "gyneco-obstetrical, death, readmission, hospitalisation >14 days)",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "compl_any_reldate",
        "field_type": "number",
        "field_label": "Number of days from admission to any complication",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_lesion",
        "field_type": "checkbox",
        "field_label": "Baseline lesion location",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_lesion___genital",
        "field_type": "binary",
        "field_label": "Genital area",
        "answer_options": None,
        "parent": "D1_lesion",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_lesion___limbs",
        "field_type": "binary",
        "field_label": "Limbs",
        "answer_options": None,
        "parent": "D1_lesion",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_lesion___face",
        "field_type": "binary",
        "field_label": "Face",
        "answer_options": None,
        "parent": "D1_lesion",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_lesion___torso",
        "field_type": "binary",
        "field_label": "Torso",
        "answer_options": None,
        "parent": "D1_lesion",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_lesioncount",
        "field_type": "number",
        "field_label": "Baseline lesion count",
        "answer_options": None,
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_severity",
        "field_type": "categorical",
        "field_label": "Baseline severity",
        "answer_options": "Critical, Severe, Moderate, Mild, None, Unknown",
        "parent": None,
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_severity___critical",
        "field_type": "binary",
        "field_label": "Critical",
        "answer_options": None,
        "parent": "D1_severity",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_severity___severe",
        "field_type": "binary",
        "field_label": "Severe",
        "answer_options": None,
        "parent": "D1_severity",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_severity___moderate",
        "field_type": "binary",
        "field_label": "Moderate",
        "answer_options": None,
        "parent": "D1_severity",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_severity___mild",
        "field_type": "binary",
        "field_label": "Mild",
        "answer_options": None,
        "parent": "D1_severity",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_severity___none",
        "field_type": "binary",
        "field_label": None,
        "answer_options": None,
        "parent": "D1_severity",
        "branching_logic": None,
        "section": None
    },
    {
        "field_name": "D1_severity___unknown",
        "field_type": "binary",
        "field_label": "Unknown",
        "answer_options": None,
        "parent": "D1_severity",
        "branching_logic": None,
        "section": None
    },
]
for d in days[1:]:
    dictionary += [
        {
            "field_name": f"D{d}_lesion",
            "field_type": "checkbox",
            "field_label": f"Lesion location on day {d}",
            "answer_options": None,
            "parent": None,
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_lesion___genital",
            "field_type": "binary",
            "field_label": "Genital area",
            "answer_options": None,
            "parent": f"D{d}_lesion",
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_lesion___limbs",
            "field_type": "binary",
            "field_label": "Limbs",
            "answer_options": None,
            "parent": f"D{d}_lesion",
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_lesion___face",
            "field_type": "binary",
            "field_label": "Face",
            "answer_options": None,
            "parent": f"D{d}_lesion",
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_lesion___torso",
            "field_type": "binary",
            "field_label": "Torso",
            "answer_options": None,
            "parent": f"D{d}_lesion",
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_lesioncount",
            "field_type": "number",
            "field_label": f"Lesion count on day {d}",
            "answer_options": None,
            "parent": None,
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_severity",
            "field_type": "categorical",
            "field_label": f"Severity on day {d}",
            "answer_options": "Critical, Severe, Moderate, Mild, None, Unknown",
            "parent": None,
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_severity___critical",
            "field_type": "binary",
            "field_label": "Critical",
            "answer_options": None,
            "parent": f"D{d}_severity",
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_severity___severe",
            "field_type": "binary",
            "field_label": "Severe",
            "answer_options": None,
            "parent": f"D{d}_severity",
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_severity___moderate",
            "field_type": "binary",
            "field_label": "Moderate",
            "answer_options": None,
            "parent": f"D{d}_severity",
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_severity___mild",
            "field_type": "binary",
            "field_label": "Mild",
            "answer_options": None,
            "parent": f"D{d}_severity",
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_severity___none",
            "field_type": "binary",
            "field_label": None,
            "answer_options": None,
            "parent": f"D{d}_severity",
            "branching_logic": None,
            "section": None
        },
        {
            "field_name": f"D{d}_severity___unknown",
            "field_type": "binary",
            "field_label": "Unknown",
            "answer_options": None,
            "parent": f"D{d}_severity",
            "branching_logic": None,
            "section": None
        },
    ]

pd.DataFrame.from_dict(dictionary).to_csv(
    os.path.join(OUTPUT_FILEPATH, "vertex_dictionary.csv"), index=False)
