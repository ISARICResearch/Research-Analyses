#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import json
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter

# =============================================================================

# Change this to match OUTPUT_FILEPATH of data_processing.py
FILEPATH = "."

# This is where the folder of VERTEX files will be saved
OUTPUT_FILEPATH = "."

# =============================================================================

tables_and_figures_filepath = os.path.join(OUTPUT_FILEPATH, "tables_and_figures/")

if not os.path.exists(tables_and_figures_filepath):
    os.makedirs(tables_and_figures_filepath)

na_values = [
    "",
    " ",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    # "None",
    "n/a",
    "nan",
    "null"
]
df_map = pd.read_csv(
    os.path.join(FILEPATH, "df_map.csv"),
    dtype={"subjid": "str"}, keep_default_na=False, na_values=na_values)
dictionary = pd.read_csv(
    os.path.join(FILEPATH, "vertex_patient_level_data/vertex_dictionary.csv"))
dictionary = dictionary.fillna("")

days = list(range(1, 15)) + [21, 28, 60, 120, 180, 360]

# =============================================================================
# Table


def get_variables_by_section_and_type(
        df, dictionary,
        required_variables=None,
        include_sections=["demog"],
        include_types=["binary", "categorical", "number"],
        exclude_suffix=[
            "_units", "addi", "otherl2", "item", "_oth",
            "_unlisted", "otherl3"],
        include_subjid=False):
    """
    Get all variables in the dataframe from specified sections and types,
    plus any required variables.
    """
    # include_ind = dictionary["field_name"].apply(
    #     lambda x: x.startswith(tuple(x + "_" for x in include_sections)))
    include_ind = dictionary["section"].isin(include_sections)
    include_ind &= dictionary["field_type"].isin(include_types)
    # include_ind &= (dictionary["field_name"].apply(
    #     lambda x: x.endswith(tuple("___" + x for x in exclude_suffix))) == 0)
    include_ind &= (dictionary["field_name"].apply(
        lambda x: x.endswith(tuple(x for x in exclude_suffix))) == 0)
    if isinstance(required_variables, list):
        include_ind |= dictionary["field_name"].isin(required_variables)
    if include_subjid:
        include_ind |= (dictionary["field_name"] == "subjid")
    include_variables = dictionary.loc[include_ind, "field_name"].tolist()
    include_variables = [col for col in include_variables if col in df.columns]
    return include_variables


def convert_categorical_to_onehot(
        df, dictionary, categorical_columns,
        sep="___", missing_val="nan", drop_first=False):
    """Convert categorical variables into onehot-encoded variables."""
    categorical_columns = [
        col for col in df.columns if col in categorical_columns]

    df.loc[:, categorical_columns] = (
        df[categorical_columns].fillna(missing_val)).apply(lambda x: x.apply(
            lambda y: y.lower().replace(" ", "_").replace("-", "_")))
    df = pd.get_dummies(
        df, columns=categorical_columns, prefix_sep=sep)

    for categorical_column in categorical_columns:
        onehot_columns = [
            var for var in df.columns
            if (var.split(sep)[0] == categorical_column)]
        # variable_type_dict["binary"] += onehot_columns
        df[onehot_columns] = df[onehot_columns].astype(object)
        if (categorical_column + sep + missing_val) in df.columns:
            mask = (df[categorical_column + sep + missing_val] == 1)
            df.loc[mask, onehot_columns] = np.nan
            df = df.drop(columns=[categorical_column + sep + missing_val])
        else:
            if drop_first:
                drop_column_ind = dictionary.apply(
                    lambda x: (
                        (x["parent"] == categorical_column) &
                        (x["field_name"].split(sep)[0] == categorical_column)
                    ), axis=1)
                df = df.drop(columns=[
                    dictionary.loc[drop_column_ind, "field_name"].values[0]])

    columns = [
        col for col in dictionary["field_name"].values if col in df.columns]
    columns += [
        col for col in df.columns
        if col not in dictionary["field_name"].values]
    df = df[columns]
    return df


def convert_onehot_to_categorical(
        df, dictionary, categorical_columns, sep="___", missing_val="nan"):
    """Convert onehot-encoded variables into categorical variables."""
    df = pd.concat([df, pd.DataFrame(columns=categorical_columns)], axis=1)
    for categorical_column in categorical_columns:
        onehot_columns = list(
            df.columns[df.columns.str.startswith(categorical_column + sep)])
        # Preserve missingness
        df.loc[:, categorical_column + sep + missing_val] = (
            (df[onehot_columns].any(axis=1) == 0) |
            (df[onehot_columns].isna().any(axis=1)))
        with pd.option_context("future.no_silent_downcasting", True):
            df.loc[:, onehot_columns] = df[onehot_columns].fillna(False)
        onehot_columns += [categorical_column + sep + missing_val]
        df.loc[:, categorical_column] = pd.from_dummies(
            df[onehot_columns], sep=sep)
        df = df.drop(columns=onehot_columns)

    columns = [
        col for col in dictionary["field_name"].values if col in df.columns]
    columns += [
        col for col in df.columns
        if col not in dictionary["field_name"].values]
    df = df[columns]
    return df


def median_iqr_str(series, add_spaces=False, dp=1, mfw=4, min_n=3):
    if series.notna().sum() < min_n:
        output_str = "N/A"
    elif add_spaces:
        mfw_f = int(np.log10(max((series.quantile(0.75), 1)))) + 2 + dp
        output_str = "%*.*f" % (mfw_f, dp, series.median()) + " ("
        output_str += "%*.*f" % (mfw_f, dp, series.quantile(0.25)) + "-"
        output_str += "%*.*f" % (mfw_f, dp, series.quantile(0.75)) + ") | "
        output_str += "%*g" % (mfw, int(series.notna().sum()))
    else:
        output_str = "%.*f" % (dp, series.median()) + " ("
        output_str += "%.*f" % (dp, series.quantile(0.25)) + "-"
        output_str += "%.*f" % (dp, series.quantile(0.75)) + ") | "
        output_str += str(int(series.notna().sum()))
    return output_str


def mean_std_str(series, add_spaces=False, dp=1, mfw=4, min_n=3):
    if series.notna().sum() < min_n:
        output_str = "N/A"
    elif add_spaces:
        mfw_f = int(max((np.log10(series.mean(), 1)))) + 2 + dp
        output_str = "%*.*f" % (mfw_f, dp, series.mean()) + " ("
        output_str += "%*.*f" % (mfw_f, dp, series.std()) + ") | "
        output_str += "%*g" % (mfw, int(series.notna().sum()))
    else:
        output_str = "%.*f" % (dp, series.mean()) + " ("
        output_str += "%.*f" % (dp, series.std()) + ") | "
        output_str += str(int(series.notna().sum()))
    return output_str


def n_percent_str(series, add_spaces=False, dp=1, mfw=4, min_n=1):
    if series.notna().sum() < min_n:
        output_str = "N/A"
    elif add_spaces:
        output_str = "%*g" % (mfw, int(series.sum())) + " ("
        percent = 100*series.mean()
        if percent == 100:
            output_str += "100.0) | "
        else:
            output_str += "%5.*f" % (dp, percent) + ") | "
        output_str += "%*g" % (mfw, int(series.notna().sum()))
    else:
        count = int(series.sum())
        percent = 100*series.mean()
        denom = int(series.notna().sum())
        output_str = f"{str(count)} ({"%.*f" % (dp, percent)}) | {str(denom)}"
    return output_str


def format_descriptive_table_variables(dictionary, max_len=100, add_key=True, sep="___", binary_symbol="*", numeric_symbol="+"):
    name = dictionary["field_name"].apply(lambda x: "   ↳ " if sep in x else "")
    name += dictionary["field_label"]
    if add_key is True:
        field_type = (
            dictionary["field_type"]
            .map({"categorical": f" ({binary_symbol})", "binary": f" ({binary_symbol})", "number": f" ({numeric_symbol})"})
            .fillna("")
        )
        name += field_type * (dictionary["field_name"].str.contains(sep) == 0)
    return name


def get_descriptive_data(
        data, dictionary, by_column=None, include_sections=["demog"],
        include_types=["binary", "categorical", "number"],
        exclude_suffix=[
            "_units", "addi", "otherl2", "item", "_oth",
            "_unlisted", "otherl3"],
        include_subjid=False, exclude_negatives=True, sep="___"):
    df = data.copy()

    include_columns = get_variables_by_section_and_type(
        df, dictionary,
        include_types=include_types, include_subjid=include_subjid,
        include_sections=include_sections, exclude_suffix=exclude_suffix)
    if (by_column is not None) & (by_column not in include_columns):
        include_columns = [by_column] + include_columns
    df = df[include_columns].dropna(axis=1, how="all").copy()

    # Convert categorical variables to onehot-encoded binary columns
    categorical_ind = (dictionary["field_type"] == "categorical")
    columns = dictionary.loc[categorical_ind, "field_name"].tolist()
    columns = [col for col in columns if col != by_column]
    df = convert_categorical_to_onehot(
        df, dictionary, categorical_columns=columns, sep=sep)

    if (by_column is not None) & (by_column not in df.columns):
        df = convert_onehot_to_categorical(
            df, dictionary, categorical_columns=[by_column], sep=sep)

    negative_values = ("no", "never smoked")
    negative_columns = [
        col for col in df.columns
        if col.split(sep)[-1].lower() in negative_values]
    if exclude_negatives:
        df.drop(columns=negative_columns, inplace=True)

    # Remove columns with only NaN values
    df = df.dropna(axis=1, how="all")
    df.fillna({by_column: "Unknown"}, inplace=True)
    return df


def descriptive_table(
        data, dictionary, by_column=None,
        include_totals=True, column_reorder=None,
        include_raw_variable_name=False, sep="___"):
    """
    Descriptive table for binary (including onehot-encoded categorical) and
    numerical variables in data. The descriptive table will have seperate
    columns for each category that exists for the variable "by_column", if
    this is provided.
    """
    df = data.copy()

    numeric_ind = (dictionary["field_type"] == "number")
    numeric_columns = dictionary.loc[numeric_ind, "field_name"].tolist()
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    binary_ind = (dictionary["field_type"] == "binary")
    binary_columns = dictionary.loc[binary_ind, "field_name"].tolist()
    binary_columns = [col for col in binary_columns if col in df.columns]

    # Add columns for section headers and categorical questions
    index = numeric_columns + binary_columns
    index += dictionary.loc[(
        dictionary["field_name"].isin(index)), "parent"].tolist()
    table_dictionary = dictionary.loc[(dictionary["field_name"].isin(index))]
    index = table_dictionary["field_name"].tolist()

    table_columns = ["Variable", "All"]
    if by_column is not None:
        add_columns = list(df[by_column].unique())
        if column_reorder is not None:
            table_columns += [
                col for col in column_reorder if col in add_columns]
            table_columns += [
                col for col in add_columns if col not in column_reorder]
        else:
            table_columns += add_columns
    table_columns += ["Raw variable name"]
    table = pd.DataFrame("", index=index, columns=table_columns)

    table["Raw variable name"] = [
        var if var in df.columns else "" for var in index]
    table["Variable"] = format_descriptive_table_variables(table_dictionary, sep=sep).tolist()

    mfw = int(np.log10(df.shape[0])) + 1  # Min field width, for formatting
    table.loc[numeric_columns, "All"] = df[numeric_columns].apply(
        median_iqr_str, mfw=mfw)
    table.loc[binary_columns, "All"] = df[binary_columns].apply(
        n_percent_str, mfw=mfw)

    totals = pd.DataFrame(columns=table_columns, index=["totals"])
    totals["Variable"] = "Totals"
    totals["All"] = df.shape[0]

    if by_column is not None:
        choices_values = df[by_column].unique()
        for value in choices_values:
            ind = (df[by_column] == value)
            mfw = int(np.log10(ind.sum())) + 1  # Min field width, for format
            table.loc[numeric_columns, value] = (
                df.loc[ind, numeric_columns].apply(median_iqr_str, mfw=mfw))
            table.loc[binary_columns, value] = (
                df.loc[ind, binary_columns].apply(n_percent_str, mfw=mfw))
            totals[value] = ind.sum()

    table = table.reset_index(drop=True)
    if include_totals:
        table = pd.concat([totals, table], axis=0).reset_index(drop=True)
    table_key = "KEY<br>(*) Count (%) | N<br>(+) Median (IQR) | N"
    if include_raw_variable_name is False:
        table.drop(columns=["Raw variable name"], inplace=True)
    return table, table_key


df_table = get_descriptive_data(
    df_map, dictionary, by_column=None,
    include_sections=dictionary["section"].unique().tolist())
# df_table.rename(columns={
#     "outco_outcome___discharged__alive": "outco_outcome___discharged_alive"},
#     inplace=True)

include_columns = [
    "demog_sex___male",
    "demog_sex___female",
    "demog_agegroup___0_4",
    "demog_agegroup___5_17",
    "demog_agegroup___18_79",
    "outco_outcome___discharged_alive",
    "outco_outcome___died",
    "outco_lesion_resolution",
    "outco_lesion_resolution_reldate",
    "comor_hiv",
    "compl_eye",
    "compl_psychological",
    "compl_gynobs",
    "compl_readmitted",
    "compl_extendedhosp",
    "compl_any",
    "preg_pregnant",
    "preg_outcome___live_birth",
    "preg_outcome___termination",
    "preg_outcome___neonatal_death",
    "expo14_type___sexual_transmission",
    "expo14_type___community_contact",
    "expo14_type___vertical_transmission",
    "expo14_type___unknown_/_other",
    "onset_firstsym___fever",
    "onset_firstsym___rash",
    "D1_lesion___genital",
    "D1_lesion___limbs",
    "D1_lesion___face",
    "D1_lesion___torso",
    "D1_severity___critical",
    "D1_severity___severe",
    "D1_severity___moderate",
    "D1_severity___mild",
    "D1_lesioncount"
]
table, table_key = descriptive_table(
    df_table[include_columns], dictionary, by_column=None,
    column_reorder=None)

table.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_table_data___0.csv"), index=False)


table_metadata = {
    "fig_id": "tables_and_figures/fig_table",
    "fig_name": "fig_table",
    "fig_arguments": {
        "table_key": "",
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": None,
        "graph_label": "Table 1",
        "graph_about":
            "Demographic summary of different variables used to"
            "\nshowcase different characteristics of the disease caused by mpox clade Ib."
            "\n(*) Some participants reported more than one location."
    },
    "fig_data": ["tables_and_figures/fig_table_data___0.csv"]
}

with open(os.path.join(tables_and_figures_filepath, "fig_table_metadata.json"), "w") as file:
    file.write(json.dumps(table_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_table_metadata.txt"), "w") as file:
    file.write(json.dumps(table_metadata))


# =============================================================================
# Heatmap


columns = [f"D{d}" for d in days]

community_ind = (df_map["expo14_type"] == "Community contact")
sexual_ind = (df_map["expo14_type"] == "Sexual transmission")

genital_columns = [x + "_lesion___genital" for x in columns]
limbs_columns = [x + "_lesion___limbs" for x in columns]
face_columns = [x + "_lesion___face" for x in columns]
torso_columns = [x + "_lesion___torso" for x in columns]

# Sexual transmission
df_heatmap = pd.DataFrame(columns=[f"Day {d}" for d in days])
df_heatmap.loc["Genital area"] = (
    df_map.loc[sexual_ind, genital_columns].sum(axis=0).tolist())
df_heatmap.loc["Limbs"] = (
    df_map.loc[sexual_ind, limbs_columns].sum(axis=0).tolist())
df_heatmap.loc["Face"] = (
    df_map.loc[sexual_ind, face_columns].sum(axis=0).tolist())
df_heatmap.loc["Torso"] = (
    df_map.loc[sexual_ind, torso_columns].sum(axis=0).tolist())
df_heatmap = (df_heatmap / sexual_ind.sum() * 100)

df_heatmap.reset_index().to_csv(
    os.path.join(tables_and_figures_filepath, "fig_heatmaps_data___0.csv"),
    index=False)

# Community contact
df_heatmap = pd.DataFrame(columns=[f"Day {d}" for d in days])
df_heatmap.loc["Genital area"] = (
    df_map.loc[community_ind, genital_columns].sum(axis=0).tolist())
df_heatmap.loc["Limbs"] = (
    df_map.loc[community_ind, limbs_columns].sum(axis=0).tolist())
df_heatmap.loc["Face"] = (
    df_map.loc[community_ind, face_columns].sum(axis=0).tolist())
df_heatmap.loc["Torso"] = (
    df_map.loc[community_ind, torso_columns].sum(axis=0).tolist())
df_heatmap = (df_heatmap / community_ind.sum() * 100)

df_heatmap.reset_index().to_csv(
    os.path.join(tables_and_figures_filepath, "fig_heatmaps_data___1.csv"),
    index=False)

heatmap_metadata = {
    "fig_id": "tables_and_figures/fig_heatmaps",
    "fig_name": "fig_heatmaps",
    "fig_arguments": {
        "title": "Transmission Routes and Lesion Locations",
        "subplot_titles": ["Sexual Transmission", "Community Contact"],
        "ylabel": "",
        "xlabel": "Days since admission",
        "colorbar_label": "Percentage<br>of patients",
        "zmin": 0,
        "zmax": 100,
        "height": 500,
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": None,
        "graph_label": "Figure 2",
        "graph_about":
            "A heatmap representing the relationship between transmission "
            "routes and rash locations over time."
    },
    "fig_data": [
        "tables_and_figures/fig_heatmaps_data___0.csv",
        "tables_and_figures/fig_heatmaps_data___1.csv"
    ]
}

with open(os.path.join(tables_and_figures_filepath, "fig_heatmaps_metadata.json"), "w") as file:
    file.write(json.dumps(heatmap_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_heatmaps_metadata.txt"), "w") as file:
    file.write(json.dumps(heatmap_metadata))

# =============================================================================
# Sankey

D1_columns = [
    "D1_lesion___face",
    "D1_lesion___torso",
    "D1_lesion___limbs",
    "D1_lesion___genital"
]
rename_dict = dict(zip(
    dictionary["field_name"], dictionary["field_label"]))
D1_labels = [rename_dict[x] for x in D1_columns]
df_map["D1_lesion_cat"] = df_map[D1_columns].apply(
    lambda x: ",".join([y for x, y in zip(x, D1_labels) if x]),
    axis=1)
df_map["D1_lesion_cat"] = df_map["D1_lesion_cat"].apply(
    lambda x: "None" if (x == "") else ("Whole body" if ("," in x) else x)
)

new_variable_dict = {
    "field_name": [
        "D1_lesion_cat",
        "D1_lesion_cat___whole_body",
        "D1_lesion_cat___face",
        "D1_lesion_cat___limbs",
        "D1_lesion_cat___torso",
        "D1_lesion_cat___genital_area",
        "D1_lesion_cat___none"
    ],
    "field_type": ["categorical"] + ["binary"]*6,
    "field_label": [
        "Baseline lesion location",
        "Whole body",
        "Face",
        "Limbs",
        "Torso",
        "Genital area",
        "None"
    ],
    "parent": [""] + ["D1_lesion_cat"]*6,
    "section": "",
    "branching_logic": "",
    "answer_options": [
        "Whole body, Face, Limbs, Torso, Genital area, None"] + [""]*6
}
dictionary = pd.concat(
    [dictionary, pd.DataFrame.from_dict(new_variable_dict)], axis=0)
dictionary = dictionary.reset_index(drop=True)

sankey_columns = [
    "demog_agegroup",
    "expo14_type",
    "D1_lesion_cat",
    "D1_severity",
    "outco_outcome"
]

color_dict_age_group = {
    "18-79": "rgb(49, 163, 84)",
    "5-17": "rgb(161, 217, 155)",
    "0-4": "rgb(229, 245, 224)",
    "Unknown": "rgb(190,190,190)"
}
color_dict_transmission = {
    "Sexual transmission": "rgb(242, 240, 247)",
    "Community contact": "rgb(158, 154, 200)",
    "Vertical transmission": "rgb(80, 31, 139)",
    "Unknown / other": "rgb(190,190,190)",
}
transmission_groups = list(color_dict_transmission.keys())
color_dict_outcome = {
    "Died": "rgb(223, 0, 105)",
    "Discharged alive": "rgb(0, 194, 111)",
    "Still hospitalised": "rgb(255, 245, 0)",
}
color_dict_severity = {
    "Critical": "rgb(118, 42, 131)",
    "Severe": "rgb(175, 141, 195)",
    "Moderate": "rgb(231, 212, 232)",
    "Mild": "rgb(217, 240, 211)",
    "None": "rgb(27, 120, 55)",
    "Unknown": "rgb(190,190,190)",
}
color_dict_rash_location = {
    "Whole body": "rgb(8, 81, 156)",
    "Genital area": "rgb(49, 130, 189)",
    "Limbs": "rgb(107, 174, 214)",
    "Face": "rgb(158, 202, 225)",
    "Torso": "rgb(198, 219, 239)",
    "None": "rgb(239, 243, 255)",
}

sort_values_dict = {
    "demog_agegroup": list(color_dict_age_group.keys()),
    "expo14_type": list(color_dict_transmission.keys()),
    "D1_lesion_cat": list(color_dict_rash_location.keys()),
    "D1_severity": list(color_dict_severity.keys()),
    "outco_outcome": list(color_dict_outcome.keys())}

sorter_key_list = [
    [y + "___" + x for x in sort_values_dict[y]]
    for y in sort_values_dict.keys()]

sorter_dict = dict(zip(
    sum(sorter_key_list, []),
    sum([list(range(len(x))) for x in sort_values_dict.values()], [])))

cmap_dict = {
    "demog_agegroup": color_dict_age_group,
    "expo14_type": color_dict_transmission,
    "D1_lesion_cat": color_dict_rash_location,
    "D1_severity": color_dict_severity,
    "outco_outcome": color_dict_outcome}

columns = [
    "demog_agegroup",
    "expo14_type",
    "D1_lesion_cat",
    "D1_severity",
    "outco_outcome"
]

df_sankey = df_map[columns].copy()
df_sankey.loc[(
    df_sankey["expo14_type"] == "Unknown"), "expo14_type"] = "Unknown / other"
df_sankey["demog_agegroup"] = df_sankey["demog_agegroup"].fillna("Unknown")
df_sankey["D1_severity"] = (
    df_sankey["D1_severity"].fillna("Unknown"))


def get_source_target(df, source, target, sorter_dict):  # from ISARIC VERTEX code
    df_new = df.groupby([source, target], observed=True).size()
    df_new = df_new.reset_index()
    df_new[source] = df_new[source].map(lambda x: source + "___" + x)
    df_new[target] = df_new[target].map(lambda x: target + "___" + x)
    df_new = df_new.sort_values(
        by=[source, target], key=lambda z: z.map(sorter_dict))
    df_new = df_new.rename(
        columns={source: "source", target: "target", 0: "value"})
    source_y = df_new.groupby("source")["value"].sum().rename("source_y")
    source_y = source_y.sort_index(key=lambda z: z.map(sorter_dict))
    source_y = (
        source_y.shift().fillna(0).cumsum() + 0.5*source_y) / source_y.sum()
    target_y = df_new.groupby("target")["value"].sum().rename("target_y")
    target_y = target_y.sort_index(key=lambda z: z.map(sorter_dict))
    target_y = (
        target_y.shift().fillna(0).cumsum() + 0.5*target_y) / target_y.sum()
    df_new = pd.merge(df_new, source_y, on="source", how="left")
    df_new = pd.merge(df_new, target_y, on="target", how="left")
    return df_new


def get_node_link(df_sankey, sorter_dict, cmap_dict, hide_labels=False):  # from ISARIC VERTEX code
    source_columns = ["source", "source_x", "source_y"]
    sankey_source = df_sankey[source_columns].rename(
        columns=dict(zip(source_columns, ["node", "x", "y"])))
    target_columns = ["target", "target_x", "target_y"]
    sankey_target = df_sankey[target_columns].rename(
        columns=dict(zip(target_columns, ["node", "x", "y"])))
    sankey_target = sankey_target.sort_values(
        by="node", key=lambda z: z.map(sorter_dict))
    nodes = pd.concat([sankey_source, sankey_target])
    nodes = nodes.loc[nodes.duplicated() == 0].reset_index(drop=True)
    nodes["variable"] = nodes["node"].apply(lambda x: x.split("___")[0])
    nodes["value"] = nodes["node"].apply(
        lambda x: "___".join(x.split("___")[1:]))

    nodes["color"] = nodes["node"].map({
        var + "___" + val: col
        for var in cmap_dict.keys()
        for val, col in cmap_dict[var].items()})

    nodes = nodes.reset_index().set_index("node")

    tol = 0.01
    nodes["x"] = nodes["x"]*(1 - 2*tol) + tol
    nodes["y"] = nodes["y"]*(1 - 2*tol) + tol

    hovertemplate_node = "%{customdata}"
    hovertemplate_link = "%{source.customdata} to %{target.customdata}"

    node = {
        "x": nodes["x"].tolist(),
        "y": nodes["y"].tolist(),
        "color": nodes["color"].tolist(),
        "customdata": [x.split("___")[1].split("_")[0] for x in nodes.index],
        "label": [x.split("___")[1] for x in nodes.index],
        "hovertemplate": hovertemplate_node,
        "pad": 15,
        "thickness": 20,
        "line": {"color": "black", "width": 1.2}}
    if hide_labels:
        node["label"] = [""] * len(node["label"])

    link_alpha = 0.3
    link = {
        "source": nodes.loc[df_sankey["source"], "index"].tolist(),
        "target": nodes.loc[df_sankey["target"], "index"].tolist(),
        "color": nodes.loc[df_sankey["source"], "color"].apply(
            lambda x: f"rgba{x[3:-1]}, 0.2)").tolist(),
        "value": df_sankey["value"].tolist(),
        "hovertemplate": hovertemplate_link,
        "line": {"color": "rgba(0,0,0," + str(link_alpha) + ")", "width": 0.3}
    }

    annotations = []
    text_x = nodes.loc[(
        nodes[["variable", "x"]].duplicated() == 0), ["variable", "x"]]
    for ind in text_x.index:
        text = text_x.loc[ind, "variable"].replace("_", " ").title()
        text = "<b>" + text + "</b>"
        x = text_x.loc[ind, "x"]
        annotations.append(dict(
            text=text, x=x, y=1.1,
            showarrow=False, xanchor="center"))
    return node, link, annotations


sankey_values_list = [
    get_source_target(df_sankey, x, y, sorter_dict)
    for x, y in zip(df_sankey.columns[:-1], df_sankey.columns[1:])]
sankey_values = pd.concat(sankey_values_list)
sankey_values["source_x"] = (sankey_values.index == 0).cumsum()
sankey_values["target_x"] = (
    sankey_values["source_x"] / sankey_values["source_x"].max())
sankey_values["source_x"] = (
    (sankey_values["source_x"] - 1) / sankey_values["source_x"].max())
sankey_values = sankey_values.reset_index(drop=True)

node, link, annotations = get_node_link(
    sankey_values, sorter_dict, cmap_dict)

annotations[0]["text"] = "<b>Age in years</b>"
annotations[1]["text"] = "<b>Transmission route</b>"
annotations[2]["text"] = "<b>Baseline lesion location</b>"
annotations[3]["text"] = "<b>Severity at baseline</b>"
annotations[4]["text"] = "<b>Outcome</b>"

node_df = pd.DataFrame.from_dict({
    k: v for k, v in node.items()
    if k in ("x", "y", "color", "customdata", "label")
})
node_df.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_sankey_data___0.csv"),
    index=False)

link_df = pd.DataFrame.from_dict({
    k: v for k, v in link.items()
    if k in ("source", "target", "color", "value")
})
link_df.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_sankey_data___1.csv"),
    index=False)

pd.DataFrame.from_dict(annotations).to_csv(
    os.path.join(tables_and_figures_filepath, "fig_sankey_data___2.csv"),
    index=False)


sankey_metadata = {
    "fig_id": "tables_and_figures/fig_sankey",
    "fig_name": "fig_sankey",
    "fig_arguments": {
        "height": 500,
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": None,
        "graph_label": "Figure 1",
        "graph_about":
            "A Sankey plot of age group, transmission route, lesion location, disease severity, "
            "and outcome in Clade 1b Mpox Cases in South Kivu, DRC (n=100)). "
            "This Sankey diagram visualizes the progression from patient age groups through "
            "transmission routes, lesion locations, initial disease severity (Severity D1), "
            "and outcomes among study participants. Each column represents a category, "
            "and the flows between them illustrate the connections and transitions within the "
            "dataset. This visualization provides a comprehensive overview of the "
            "relationships between patient demographics, exposure/transmission types, "
            "clinical presentation, severity, and outcomes. For transmission routes, a "
            "ranking system was applied. For example, if patients reported both community "
            "contact and sexual transmission, they were categorized under sexual transmission."
    },
    "fig_data": [
        "tables_and_figures/fig_sankey_data___0.csv",
        "tables_and_figures/fig_sankey_data___1.csv",
        "tables_and_figures/fig_sankey_data___2.csv"
    ]
}

with open(os.path.join(tables_and_figures_filepath, "fig_sankey_metadata.json"), "w") as file:
    file.write(json.dumps(sankey_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_sankey_metadata.txt"), "w") as file:
    file.write(json.dumps(sankey_metadata))


# =============================================================================
# Severity bar chart

df_severity = pd.DataFrame(
    index=["None", "Mild", "Moderate", "Severe", "Critical"])

for column in [f"D{d}" for d in days]:
    df_severity = pd.merge(
        df_severity,
        df_map[column + "_severity"].value_counts().rename(column),
        left_index=True,
        right_index=True,
        how="left"
    )
df_severity = df_severity.T.reset_index()
df_severity["index"] = [f"Day {x}" for x in days]

df_severity.fillna(0).to_csv(
    os.path.join(tables_and_figures_filepath, "fig_bar_chart_data___0.csv"),
    index=False)

count = df_map["subjid"].count()

bar_chart_metadata = {
    "fig_id": "tables_and_figures/fig_bar_chart",
    "fig_name": "fig_bar_chart",
    "fig_arguments": {
        "title": f"Severity over time during observation period (all, N={count})",
        "ylabel": "Count",
        "xlabel": "Days since admission",
        "base_color_map": {
                "Critical": "rgb(118, 42, 131)",
                "Severe": "rgb(175, 141, 195)",
                "Moderate": "rgb(231, 212, 232)",
                "Mild": "rgb(217, 240, 211)",
                "None": "rgb(27, 120, 55)",
                "Unknown": "rgb(190,190,190)",
                "Discharged alive": "rgb(0, 194, 111)",
                "Died": "rgb(223, 0, 105)"
        },
        "height": 500,
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": None,
        "graph_label": "Figure 3a",
        "graph_about":
            "Longitudinal distribution of disease severity among study participants "
            "over time in South Kivu, DRC. This chart shows the distribution of "
            "disease severity across multiple time points, from Day 1 (D1) to Day 28 (D28). "
            "The chart segments the population into different severity levels—critical, "
            "severe, moderate, mild, and unknown—allowing a visualisation of changes "
            "in the severity distribution over time. Note: In this figure, the definition "
            "of severity is solely based on lesion counts, without marking infants and "
            "pregnant women as 'Severe'."
    },
    "fig_data": [
        "tables_and_figures/fig_bar_chart_data___0.csv"
    ]
}

with open(os.path.join(tables_and_figures_filepath, "fig_bar_chart_metadata.json"), "w") as file:
    file.write(json.dumps(bar_chart_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_bar_chart_metadata.txt"), "w") as file:
    file.write(json.dumps(bar_chart_metadata))


# Stratified by HIV status
df_severity = pd.DataFrame(
    index=["None", "Mild", "Moderate", "Severe", "Critical"])

for column in [f"D{d}" for d in days]:
    df_severity = pd.merge(
        df_severity,
        df_map.loc[
            df_map["comor_hiv"],
            column + "_severity"
        ].value_counts().rename(column),
        left_index=True,
        right_index=True,
        how="left"
    )
df_severity = df_severity.T.reset_index()
df_severity["index"] = [f"Day {x}" for x in days]

df_severity.fillna(0).to_csv(
    os.path.join(tables_and_figures_filepath, "fig_bar_chart_hiv_pos_data___0.csv"),
    index=False)

count = df_map["comor_hiv"].sum()

bar_chart_hiv_pos_metadata = {
    "fig_id": "tables_and_figures/fig_bar_chart_hiv_pos",
    "fig_name": "fig_bar_chart",
    "fig_arguments": {
        "title": f"Severity over time during observation period (HIV positive, N={count})",
        "ylabel": "Count",
        "xlabel": "Days since admission",
        "base_color_map": {
                "Critical": "rgb(118, 42, 131)",
                "Severe": "rgb(175, 141, 195)",
                "Moderate": "rgb(231, 212, 232)",
                "Mild": "rgb(217, 240, 211)",
                "None": "rgb(27, 120, 55)",
                "Unknown": "rgb(190,190,190)",
                "Discharged alive": "rgb(0, 194, 111)",
                "Died": "rgb(223, 0, 105)"
        },
        "height": 500,
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": "fig_bar_chart_hiv_pos",
        "graph_label": "Figure 3b",
        "graph_about":
            "Longitudinal distribution of disease severity among study participants "
            "over time in South Kivu, DRC. This chart shows the distribution of "
            "disease severity across multiple time points, from Day 1 (D1) to Day 28 (D28). "
            "The chart segments the population into different severity levels—critical, "
            "severe, moderate, mild, and unknown—allowing a visualisation of changes "
            "in the severity distribution over time. Note: In this figure, the definition "
            "of severity is solely based on lesion counts, without marking infants and "
            "pregnant women as 'Severe'."
    },
    "fig_data": [
        "tables_and_figures/fig_bar_chart_hiv_pos_data___0.csv"
    ]
}

with open(os.path.join(tables_and_figures_filepath, "fig_bar_chart_hiv_pos_metadata.json"), "w") as file:
    file.write(json.dumps(bar_chart_hiv_pos_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_bar_chart_hiv_pos_metadata.txt"), "w") as file:
    file.write(json.dumps(bar_chart_hiv_pos_metadata))


df_severity = pd.DataFrame(
    index=["None", "Mild", "Moderate", "Severe", "Critical"])

for column in [f"D{d}" for d in days]:
    df_severity = pd.merge(
        df_severity,
        df_map.loc[
            ~df_map["comor_hiv"],
            column + "_severity"
        ].value_counts().rename(column),
        left_index=True,
        right_index=True,
        how="left"
    )
df_severity = df_severity.T.reset_index()
df_severity["index"] = [f"Day {x}" for x in days]

df_severity.fillna(0).to_csv(
    os.path.join(tables_and_figures_filepath, "fig_bar_chart_hiv_neg_data___0.csv"),
    index=False)

count = (~df_map["comor_hiv"]).sum()

bar_chart_hiv_neg_metadata = {
    "fig_id": "tables_and_figures/fig_bar_chart_hiv_pos",
    "fig_name": "fig_bar_chart",
    "fig_arguments": {
        "title": f"Severity over time during observation period (HIV negative, N={count})",
        "ylabel": "Count",
        "xlabel": "Days since admission",
        "base_color_map": {
                "Critical": "rgb(118, 42, 131)",
                "Severe": "rgb(175, 141, 195)",
                "Moderate": "rgb(231, 212, 232)",
                "Mild": "rgb(217, 240, 211)",
                "None": "rgb(27, 120, 55)",
                "Unknown": "rgb(190,190,190)",
                "Discharged alive": "rgb(0, 194, 111)",
                "Died": "rgb(223, 0, 105)"
        },
        "height": 500,
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": "fig_bar_chart_hiv_pos",
        "graph_label": "Figure 3c",
        "graph_about":
            "Longitudinal distribution of disease severity among study participants "
            "over time in South Kivu, DRC. This chart shows the distribution of "
            "disease severity across multiple time points, from Day 1 (D1) to Day 28 (D28). "
            "The chart segments the population into different severity levels—critical, "
            "severe, moderate, mild, and unknown—allowing a visualisation of changes "
            "in the severity distribution over time. Note: In this figure, the definition "
            "of severity is solely based on lesion counts, without marking infants and "
            "pregnant women as 'Severe'."
    },
    "fig_data": [
        "tables_and_figures/fig_bar_chart_hiv_neg_data___0.csv"
    ]
}

with open(os.path.join(tables_and_figures_filepath, "fig_bar_chart_hiv_neg_metadata.json"), "w") as file:
    file.write(json.dumps(bar_chart_hiv_neg_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_bar_chart_hiv_neg_metadata.txt"), "w") as file:
    file.write(json.dumps(bar_chart_hiv_neg_metadata))


# # =============================================================================
# # Kaplan-Meier
#
#
# def execute_kaplan_meier(  # adapted from ISARIC VERTEX code
#     df,
#     duration_col,
#     event_col,
#     group_col,
#     at_risk_times,
#     alpha=0.05,
#     c_prob=False,
# ):
#     # Remove rows with missing values in relevant columns
#     df = df.dropna(subset=[duration_col, event_col, group_col])
#     kmf = KaplanMeierFitter()
#
#     unique_groups = df[group_col].sort_values().unique()
#
#     df_km = pd.DataFrame(columns=["timeline"])
#
#     # Compute survival curves and confidence intervals for each group
#     for group in unique_groups:
#         group_data = df[df[group_col] == group]
#         kmf.fit(
#             group_data[duration_col],
#             event_observed=group_data[event_col],
#             label=str(group),
#             alpha=alpha)
#         ci_lower = kmf.confidence_interval_[f"{group}_lower_{1 - alpha}"] * 100
#         ci_upper = kmf.confidence_interval_[f"{group}_upper_{1 - alpha}"] * 100
#         survival_curve = pd.concat(
#             [kmf.survival_function_ * 100, ci_lower, ci_upper], axis=1)
#         if c_prob:
#             survival_curve = 100 - survival_curve
#         survival_curve = survival_curve.drop_duplicates().reset_index().rename(
#             columns={"index": "timeline"})
#         df_km = pd.merge(
#             df_km, survival_curve, on="timeline", how="outer").bfill()
#
#     # Generate risk table: number of individuals at risk over time
#     risk_counts = {
#         group: [(
#             (df[group_col] == group) & (df[duration_col] >= t)).sum()
#             for t in at_risk_times]
#         for group in unique_groups
#     }
#
#     risk_table = pd.DataFrame(risk_counts, index=at_risk_times).T
#     risk_table.insert(0, "Group", risk_table.index)
#
#     return df_km, risk_table
#
#
# # Kaplan Meier to D28, censoring complications by D14
# km_columns = ["comor_hiv", "outco_lesion_resolution", "outco_lesion_resolution_reldate"]
# df_km = df_map[km_columns].rename(
#     columns={"outco_lesion_resolution_reldate": "outco_time_to_event"},
# ).copy()
# df_km.loc[
#     df_map["compl_any"]
#     & (df_map["compl_any_reldate"] < 14),
#     "outco_lesion_resolution"
# ] = False
# df_km.loc[
#     df_map["compl_any"]
#     & (df_map["compl_any_reldate"] < 14),
#     "outco_time_to_event"
# ] = df_map.loc[
#     df_map["compl_any"]
#     & (df_map["compl_any_reldate"] < 14),
#     "compl_any_reldate"
# ]
#
# # Administrative censoring on D28
# df_km.loc[
#     df_map["outco_lesion_resolution_reldate"] >= 28,
#     "outco_lesion_resolution"
# ] = False
# df_km.loc[
#     df_map["outco_lesion_resolution_reldate"] >= 28,
#     "outco_time_to_event"
# ] = 28
#
# df_km.loc[
#     ~df_km["outco_lesion_resolution"],
#     "outco_time_to_event"
# ] = df_km.loc[~df_km["outco_lesion_resolution"], "outco_time_to_event"].fillna(28)
#
# df_km["comor_hiv"] = df_km["comor_hiv"].replace({True: "HIV positive", False: "HIV negative"})
#
# km_table, at_risk_table = execute_kaplan_meier(
#     df_km,
#     duration_col="outco_time_to_event",
#     event_col="outco_lesion_resolution",
#     group_col="comor_hiv",
#     c_prob=True,
#     at_risk_times=[0, 3, 7, 14, 21, 28]
# )
#
# km_table.to_csv(
#     os.path.join(tables_and_figures_filepath, "fig_kaplan_meier_data___0.csv"),
#     index=False)
# at_risk_table.to_csv(
#     os.path.join(tables_and_figures_filepath, "fig_kaplan_meier_data___1.csv"),
#     index=False)
#
#
# kaplan_meier_metadata = {
#     "fig_id": "tables_and_figures/fig_kaplan_meier",
#     "fig_name": "fig_kaplan_meier",
#     "fig_arguments": {
#         "title": "Kaplan-Meier curves for lesion resolution",
#         "ylabel": "Cumulative Probability of Lesion Resolution (%)",
#         "xlabel": "Time from admission (days)",
#         "index_column": "Group",
#         "xlim": [0, 50],
#         "height": 700,
#         "suffix": "tables_and_figures",
#         "filepath": "",
#         "save_inputs": False,
#         "graph_id": None,
#         "graph_label": "Figure 4",
#         "graph_about":
#             "Kaplan-Meier survival curve illustrating the probability "
#             "of lesion resolution over time, stratified by HIV status "
#             "(HIV positive and HIV negative). The x-axis represents "
#             "the days to recovery (onset to resolution), "
#             "while the y-axis shows the cumulative probability of lesion resolution "
#             "without complicaitons. Shaded areas "
#             "around each curve indicate the 95% confidence intervals. "
#             "Numbers below the x-axis represent the participants at risk "
#             "in each age group at specific time points."
#     },
#     "fig_data": [
#         "tables_and_figures/fig_kaplan_meier_data___0.csv",
#         "tables_and_figures/fig_kaplan_meier_data___1.csv"
#     ]
# }
#
#
# with open(os.path.join(tables_and_figures_filepath, "fig_kaplan_meier_metadata.json"), "w") as file:
#     file.write(json.dumps(kaplan_meier_metadata))
#
# with open(os.path.join(tables_and_figures_filepath, "fig_kaplan_meier_metadata.txt"), "w") as file:
#     file.write(json.dumps(kaplan_meier_metadata))

# =============================================================================
# Survival models

# Cox model for time to lesion resolution without complications by D14
# administrative censoring at D28

covariates = [
    "demog_agegroup___0_4",
    "demog_agegroup___5_17",
    "demog_sex___male",
    "expo14_type___vertical_transmission",
    "expo14_type___sexual_transmission",
    "onset_firstsym___rash",
    "D1_lesion___genital",
    "D1_lesion___torso",
    "D1_lesion___limbs",
    "D1_lesion___face",
    "D1_lesioncount",
    "comor_hiv",
]

df_cox = df_map.rename(
    columns={"outco_lesion_resolution_reldate": "outco_time_to_event"},
).copy()
df_cox.loc[
    df_cox["compl_any"]
    & (df_cox["compl_any_reldate"] < 14),
    "outco_lesion_resolution"
] = False
df_cox.loc[
    df_cox["compl_any"]
    & (df_cox["compl_any_reldate"] < 14),
    "outco_time_to_event"
] = df_cox.loc[
    df_cox["compl_any"]
    & (df_cox["compl_any_reldate"] < 14),
    "compl_any_reldate"
]

# Administrative censoring on D28
df_cox.loc[
    df_cox["outco_time_to_event"] >= 28,
    "outco_lesion_resolution"
] = False
df_cox.loc[
    df_cox["outco_time_to_event"] >= 28,
    "outco_time_to_event"
] = 28

df_cox.loc[
    ~df_cox["outco_lesion_resolution"],
    "outco_time_to_event"
] = df_cox.loc[
    ~df_cox["outco_lesion_resolution"],
    "outco_time_to_event"
].fillna(28)

df_cox = pd.concat(
    [
        df_cox[["subjid", "outco_time_to_event", "outco_lesion_resolution"]],
        df_table[covariates]
    ],
    axis=1,
).dropna(subset="outco_time_to_event")

cph = CoxPHFitter()
cph.fit(
    df_cox.drop(columns=["subjid"]),
    duration_col="outco_time_to_event",
    event_col="outco_lesion_resolution"
)
cph.check_assumptions(
    df_cox.drop(columns=["subjid"]),
    p_value_threshold=0.05,
    show_plots=False,
)

# After model checking, try with model stratifying by age and first symptom removed
# Also there is very little vertical transmission, so it would make sense to
# merge vertical transmission and community contact into a single category

df_cox.drop(
    columns=[
        "demog_agegroup___0_4",
        "demog_agegroup___5_17",
        "expo14_type___vertical_transmission",
        "onset_firstsym___rash",
    ],
    inplace=True
)
df_cox = pd.merge(
    df_cox,
    df_map[["subjid", "demog_agegroup"]],
    on="subjid",
    how="left",
)

# Lesion count in hundreds, doesn't change model but better interpretability
df_cox["D1_lesioncount"] = df_cox["D1_lesioncount"] / 100

cph.fit(
    df_cox.drop(columns=["subjid"]),
    duration_col="outco_time_to_event",
    event_col="outco_lesion_resolution",
    strata=["demog_agegroup"],
)
cph.check_assumptions(
    df_cox.drop(columns=["subjid"]),
    p_value_threshold=0.05,
    show_plots=False
)

# Cause-specific Cox models for two causes: lesion resolution and complications
# Model the hazard for each cause while patient is event-free, so right censor if
# another cause if observed
# No D28 administrative censoring and no restriction about complication date
# Use same covariates/strata

covariates = [
    "demog_agegroup",
    "demog_sex___male",
    "expo14_type___sexual_transmission",
    "comor_hiv",
    "D1_lesion___genital",
    "D1_lesion___torso",
    "D1_lesion___limbs",
    "D1_lesion___face",
    "D1_lesioncount",
]

df_cs_cox = df_map.rename(
    columns={
        "outco_lesion_resolution": "cause",
        "outco_lesion_resolution_reldate": "outco_time_to_event",
    },
).copy()
df_cs_cox["cause"] = df_cs_cox["cause"].astype(int)
idx = (
    (df_cs_cox["outco_time_to_event"] > df_map["compl_any_reldate"])
    | (df_cs_cox["outco_time_to_event"].isna() & df_cs_cox["compl_any_reldate"].notna())
)

df_cs_cox.loc[idx, "cause"] = 2
df_cs_cox.loc[idx, "outco_time_to_event"] = df_cs_cox.loc[idx, "compl_any_reldate"]

df_cs_cox = pd.concat(
    [
        df_cs_cox[[
            "subjid",
            "outco_time_to_event",
            "cause",
            "demog_agegroup",
            "comor_hiv",
            "D1_lesion___genital",
            "D1_lesion___torso",
            "D1_lesion___limbs",
            "D1_lesion___face",
            "D1_lesioncount",
        ]],
        df_table[[
            "demog_sex___male",
            "expo14_type___sexual_transmission",
        ]],
    ],
    axis=1,
).dropna(subset="outco_time_to_event")

# Lesion count in hundreds, doesn't change model but better interpretability
df_cs_cox["D1_lesioncount"] = df_cs_cox["D1_lesioncount"] / 100

cph_cs_resolved = CoxPHFitter()
cph_cs_resolved.fit(
    df_cs_cox.drop(columns=["subjid"]).replace({"cause": {1: True, 2: False}}),
    duration_col="outco_time_to_event",
    event_col="cause",
    strata=["demog_agegroup"],
)
cph_cs_resolved.check_assumptions(
    df_cs_cox.drop(columns=["subjid"]).replace({"cause": {1: True, 2: False}}),
    p_value_threshold=0.05,
    show_plots=False,
)  # Sexual transmission is close to threshold, judged to be ok

cph_cs_compl = CoxPHFitter()
cph_cs_compl.fit(
    df_cs_cox.drop(columns=["subjid"]).replace({"cause": {2: True, 1: False}}),
    duration_col="outco_time_to_event",
    event_col="cause",
    strata=["demog_agegroup"],
)
cph_cs_compl.check_assumptions(
    df_cs_cox.drop(columns=["subjid"]).replace({"cause": {2: True, 1: False}}),
    p_value_threshold=0.05,
    show_plots=False,
)

# Cumulative incidence estimate (using Aalen-Johansen estimator)
aj_resolved_hiv_pos = AalenJohansenFitter(calculate_variance=True)
aj_resolved_hiv_pos.fit(
    durations=df_cs_cox.loc[df_cs_cox["comor_hiv"], "outco_time_to_event"],
    event_observed=df_cs_cox.loc[df_cs_cox["comor_hiv"], "cause"],
    event_of_interest=1
)
aj_compl_hiv_pos = AalenJohansenFitter(calculate_variance=True)
aj_compl_hiv_pos.fit(
    durations=df_cs_cox.loc[df_cs_cox["comor_hiv"], "outco_time_to_event"],
    event_observed=df_cs_cox.loc[df_cs_cox["comor_hiv"], "cause"],
    event_of_interest=2
)
aj_resolved_hiv_neg = AalenJohansenFitter(calculate_variance=True)
aj_resolved_hiv_neg.fit(
    durations=df_cs_cox.loc[~df_cs_cox["comor_hiv"], "outco_time_to_event"],
    event_observed=df_cs_cox.loc[~df_cs_cox["comor_hiv"], "cause"],
    event_of_interest=1
)
aj_compl_hiv_neg = AalenJohansenFitter(calculate_variance=True)
aj_compl_hiv_neg.fit(
    durations=df_cs_cox.loc[~df_cs_cox["comor_hiv"], "outco_time_to_event"],
    event_observed=df_cs_cox.loc[~df_cs_cox["comor_hiv"], "cause"],
    event_of_interest=2
)

# Build a combined table on a common time grid
times = sorted(
    set(aj_resolved_hiv_pos.cumulative_density_.index)
    .union(aj_compl_hiv_pos.cumulative_density_.index)
    .union(aj_resolved_hiv_neg.cumulative_density_.index)
    .union(aj_compl_hiv_neg.cumulative_density_.index)
)
# Reindex (forward fill to carry last estimate forward)
c_resolved_hiv_pos = (
    aj_resolved_hiv_pos.cumulative_density_
    .reindex(times, method="ffill")
    .fillna(0)
)
c_compl_hiv_pos = (
    aj_compl_hiv_pos.cumulative_density_
    .reindex(times, method="ffill")
    .fillna(0)
)
c_resolved_hiv_neg = (
    aj_resolved_hiv_neg.cumulative_density_
    .reindex(times, method="ffill")
    .fillna(0)
)
c_compl_hiv_neg = (
    aj_compl_hiv_neg.cumulative_density_
    .reindex(times, method="ffill")
    .fillna(0)
)
ci_resolved_hiv_pos = (
    aj_resolved_hiv_pos.confidence_interval_
    .reindex(times, method="ffill")
    .fillna(0)
)
ci_compl_hiv_pos = (
    aj_compl_hiv_pos.confidence_interval_
    .reindex(times, method="ffill")
    .fillna(0)
)
ci_resolved_hiv_neg = (
    aj_resolved_hiv_neg.confidence_interval_
    .reindex(times, method="ffill")
    .fillna(0)
)
ci_compl_hiv_neg = (
    aj_compl_hiv_neg.confidence_interval_
    .reindex(times, method="ffill")
    .fillna(0)
)

aj_resolved = pd.DataFrame({
    "timeline": times,
    "HIV positive": 100*c_resolved_hiv_pos.iloc[:, 0].values,
    "HIV positive_lower_0.95": 100*ci_resolved_hiv_pos.iloc[:, 0].values,
    "HIV positive_upper_0.95": 100*ci_resolved_hiv_pos.iloc[:, 1].values,
    "HIV negative": 100*c_resolved_hiv_neg.iloc[:, 0].values,
    "HIV negative_lower_0.95": 100*ci_resolved_hiv_neg.iloc[:, 0].values,
    "HIV negative_upper_0.95": 100*ci_resolved_hiv_neg.iloc[:, 1].values,
})
aj_compl = pd.DataFrame({
    "timeline": times,
    "HIV positive": 100*c_compl_hiv_pos.iloc[:, 0].values,
    "HIV positive_lower_0.95": 100*ci_compl_hiv_pos.iloc[:, 0].values,
    "HIV positive_upper_0.95": 100*ci_compl_hiv_pos.iloc[:, 1].values,
    "HIV negative": 100*c_compl_hiv_neg.iloc[:, 0].values,
    "HIV negative_lower_0.95": 100*ci_compl_hiv_neg.iloc[:, 0].values,
    "HIV negative_upper_0.95": 100*ci_compl_hiv_neg.iloc[:, 1].values,
})

# Generate risk table: number of individuals at risk over time
at_risk_times = [0, 7, 14, 21, 28, 35, 42, 49]
at_risk_index = [
    "HIV positive, at risk:",
    "HIV positive, lesions resolved so far:",
    "HIV positive, complications so far:",
    "HIV negative, at risk:",
    "HIV negative, lesions resolved so far:",
    "HIV negative, complications so far:",
]

at_risk_table = pd.DataFrame(columns=at_risk_times, index=at_risk_index)
at_risk_table.loc["HIV positive, at risk:"] = [
    (
        df_cs_cox["comor_hiv"]
        & (df_cs_cox["outco_time_to_event"] > x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV positive, lesions resolved so far:"] = [
    (
        df_cs_cox["comor_hiv"]
        & (df_cs_cox["cause"] == 1)
        & (df_cs_cox["outco_time_to_event"] <= x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV positive, complications so far:"] = [
    (
        df_cs_cox["comor_hiv"]
        & (df_cs_cox["cause"] == 2)
        & (df_cs_cox["outco_time_to_event"] <= x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV negative, at risk:"] = [
    (
        ~df_cs_cox["comor_hiv"]
        & (df_cs_cox["outco_time_to_event"] > x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV negative, lesions resolved so far:"] = [
    (
        ~df_cs_cox["comor_hiv"]
        & (df_cs_cox["cause"] == 1)
        & (df_cs_cox["outco_time_to_event"] <= x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV negative, complications so far:"] = [
    (
        ~df_cs_cox["comor_hiv"]
        & (df_cs_cox["cause"] == 2)
        & (df_cs_cox["outco_time_to_event"] <= x)
    ).sum()
    for x in at_risk_times
]
at_risk_table = at_risk_table.reset_index().rename(columns={"index": "Group"})

max_time = 50

aj_resolved.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_resolved_data___0.csv"),
    index=False)
at_risk_table.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_resolved_data___1.csv"),
    index=False)

cumulative_incidence_resolved_metadata = {
    "fig_id": "tables_and_figures/fig_cumulative_incidence_resolved",
    "fig_name": "fig_kaplan_meier",  # This is to re-purpose existing ISARIC VERTEX code
    "fig_arguments": {
        "title": "Lesions resolved: cumulative incidence",
        "ylabel": "Cumulative Incidence (%)",
        "xlabel": "Time from admission (days)",
        "index_column": "Group",
        "xlim": [0, max_time],
        "base_color_map": {"HIV positive": "hsl(146,57,40)", "HIV negative": "hsl(204,61,45)"},
        "height": 700,
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": None,
        "graph_label": "Figure 4a",
        "graph_about":
            "Cumulative incidence of lesion resolution, stratified "
            "by HIV status (cause-specific, against complications). "
            "Estimated using Aalen-Johansen."
    },
    "fig_data": [
        "tables_and_figures/fig_cumulative_incidence_resolved_data___0.csv",
        "tables_and_figures/fig_cumulative_incidence_resolved_data___1.csv"
    ]
}

with open(os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_resolved_metadata.json"), "w") as file:
    file.write(json.dumps(cumulative_incidence_resolved_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_resolved_metadata.txt"), "w") as file:
    file.write(json.dumps(cumulative_incidence_resolved_metadata))

aj_compl.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_compl_data___0.csv"),
    index=False)
at_risk_table.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_compl_data___1.csv"),
    index=False)

cumulative_incidence_compl_metadata = {
    "fig_id": "tables_and_figures/fig_cumulative_incidence_compl",
    "fig_name": "fig_kaplan_meier",  # This is to re-purpose existing ISARIC VERTEX code
    "fig_arguments": {
        "title": "Complications: cumulative incidence",
        "ylabel": "Cumulative Incidence (%)",
        "xlabel": "Time from admission (days)",
        "index_column": "Group",
        "xlim": [0, max_time],
        "base_color_map": {"HIV positive": "hsl(19,90,50)", "HIV negative": "hsl(277,32,50)"},
        "height": 700,
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": None,
        "graph_label": "Figure 4b",
        "graph_about":
            "Cumulative incidence of complications, stratified "
            "by HIV status (cause-specific, against lesions resolved). "
            "Estimated using Aalen-Johansen."
    },
    "fig_data": [
        "tables_and_figures/fig_cumulative_incidence_compl_data___0.csv",
        "tables_and_figures/fig_cumulative_incidence_compl_data___1.csv"
    ]
}

with open(os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_compl_metadata.json"), "w") as file:
    file.write(json.dumps(cumulative_incidence_compl_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_compl_metadata.txt"), "w") as file:
    file.write(json.dumps(cumulative_incidence_compl_metadata))


# With onset of symptoms as time origin
df_cs_cox = pd.merge(
    df_cs_cox,
    df_map["dates_onset_reldate"],
    left_index=True,
    right_index=True
)
df_cs_cox["outco_time_to_event"] = (
    df_cs_cox["outco_time_to_event"]
    - df_cs_cox["dates_onset_reldate"]
)
df_cs_cox = (
    df_cs_cox
    .drop(columns=["dates_onset_reldate"])
    .dropna(subset=["outco_time_to_event"])
)

# Repeat Cox model and Aalen-Johansen
cph_cs_onset_resolved = CoxPHFitter()
cph_cs_onset_resolved.fit(
    df_cs_cox.drop(columns=["subjid"]).replace({"cause": {1: True, 2: False}}),
    duration_col="outco_time_to_event",
    event_col="cause",
    strata=["demog_agegroup"],
)
cph_cs_onset_resolved.check_assumptions(
    df_cs_cox.drop(columns=["subjid"]).replace({"cause": {1: True, 2: False}}),
    p_value_threshold=0.05,
    show_plots=False,
)

cph_cs_onset_compl = CoxPHFitter()
cph_cs_onset_compl.fit(
    df_cs_cox.drop(columns=["subjid"]).replace({"cause": {2: True, 1: False}}),
    duration_col="outco_time_to_event",
    event_col="cause",
    strata=["demog_agegroup"],
)
cph_cs_onset_compl.check_assumptions(
    df_cs_cox.drop(columns=["subjid"]).replace({"cause": {2: True, 1: False}}),
    p_value_threshold=0.05,
    show_plots=False,
)

# Cumulative incidence estimate (using Aalen-Johansen estimator)
aj_resolved_hiv_pos = AalenJohansenFitter(calculate_variance=True)
aj_resolved_hiv_pos.fit(
    durations=df_cs_cox.loc[df_cs_cox["comor_hiv"], "outco_time_to_event"],
    event_observed=df_cs_cox.loc[df_cs_cox["comor_hiv"], "cause"],
    event_of_interest=1
)
aj_compl_hiv_pos = AalenJohansenFitter(calculate_variance=True)
aj_compl_hiv_pos.fit(
    durations=df_cs_cox.loc[df_cs_cox["comor_hiv"], "outco_time_to_event"],
    event_observed=df_cs_cox.loc[df_cs_cox["comor_hiv"], "cause"],
    event_of_interest=2
)
aj_resolved_hiv_neg = AalenJohansenFitter(calculate_variance=True)
aj_resolved_hiv_neg.fit(
    durations=df_cs_cox.loc[~df_cs_cox["comor_hiv"], "outco_time_to_event"],
    event_observed=df_cs_cox.loc[~df_cs_cox["comor_hiv"], "cause"],
    event_of_interest=1
)
aj_compl_hiv_neg = AalenJohansenFitter(calculate_variance=True)
aj_compl_hiv_neg.fit(
    durations=df_cs_cox.loc[~df_cs_cox["comor_hiv"], "outco_time_to_event"],
    event_observed=df_cs_cox.loc[~df_cs_cox["comor_hiv"], "cause"],
    event_of_interest=2
)

# Build a combined table on a common time grid
times = sorted(
    set(aj_resolved_hiv_pos.cumulative_density_.index)
    .union(aj_compl_hiv_pos.cumulative_density_.index)
    .union(aj_resolved_hiv_neg.cumulative_density_.index)
    .union(aj_compl_hiv_neg.cumulative_density_.index)
)
# Reindex (forward fill to carry last estimate forward)
c_resolved_hiv_pos = (
    aj_resolved_hiv_pos.cumulative_density_
    .reindex(times, method="ffill")
    .fillna(0)
)
c_compl_hiv_pos = (
    aj_compl_hiv_pos.cumulative_density_
    .reindex(times, method="ffill")
    .fillna(0)
)
c_resolved_hiv_neg = (
    aj_resolved_hiv_neg.cumulative_density_
    .reindex(times, method="ffill")
    .fillna(0)
)
c_compl_hiv_neg = (
    aj_compl_hiv_neg.cumulative_density_
    .reindex(times, method="ffill")
    .fillna(0)
)

ci_resolved_hiv_pos = (
    aj_resolved_hiv_pos.confidence_interval_
    .reindex(times, method="ffill")
    .fillna(0)
)
ci_compl_hiv_pos = (
    aj_compl_hiv_pos.confidence_interval_
    .reindex(times, method="ffill")
    .fillna(0)
)
ci_resolved_hiv_neg = (
    aj_resolved_hiv_neg.confidence_interval_
    .reindex(times, method="ffill")
    .fillna(0)
)
ci_compl_hiv_neg = (
    aj_compl_hiv_neg.confidence_interval_
    .reindex(times, method="ffill")
    .fillna(0)
)

aj_resolved = pd.DataFrame({
    "timeline": times,
    "HIV positive": 100*c_resolved_hiv_pos.iloc[:, 0].values,
    "HIV positive_lower_0.95": 100*ci_resolved_hiv_pos.iloc[:, 0].values,
    "HIV positive_upper_0.95": 100*ci_resolved_hiv_pos.iloc[:, 1].values,
    "HIV negative": 100*c_resolved_hiv_neg.iloc[:, 0].values,
    "HIV negative_lower_0.95": 100*ci_resolved_hiv_neg.iloc[:, 0].values,
    "HIV negative_upper_0.95": 100*ci_resolved_hiv_neg.iloc[:, 1].values,
})
aj_compl = pd.DataFrame({
    "timeline": times,
    "HIV positive": 100*c_compl_hiv_pos.iloc[:, 0].values,
    "HIV positive_lower_0.95": 100*ci_compl_hiv_pos.iloc[:, 0].values,
    "HIV positive_upper_0.95": 100*ci_compl_hiv_pos.iloc[:, 1].values,
    "HIV negative": 100*c_compl_hiv_neg.iloc[:, 0].values,
    "HIV negative_lower_0.95": 100*ci_compl_hiv_neg.iloc[:, 0].values,
    "HIV negative_upper_0.95": 100*ci_compl_hiv_neg.iloc[:, 1].values,
})

# Generate risk table: number of individuals at risk over time
at_risk_table = pd.DataFrame(columns=at_risk_times, index=at_risk_index)
at_risk_table.loc["HIV positive, at risk:"] = [
    (
        df_cs_cox["comor_hiv"]
        & (df_cs_cox["outco_time_to_event"] > x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV positive, lesions resolved so far:"] = [
    (
        df_cs_cox["comor_hiv"]
        & (df_cs_cox["cause"] == 1)
        & (df_cs_cox["outco_time_to_event"] <= x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV positive, complications so far:"] = [
    (
        df_cs_cox["comor_hiv"]
        & (df_cs_cox["cause"] == 2)
        & (df_cs_cox["outco_time_to_event"] <= x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV negative, at risk:"] = [
    (
        ~df_cs_cox["comor_hiv"]
        & (df_cs_cox["outco_time_to_event"] > x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV negative, lesions resolved so far:"] = [
    (
        ~df_cs_cox["comor_hiv"]
        & (df_cs_cox["cause"] == 1)
        & (df_cs_cox["outco_time_to_event"] <= x)
    ).sum()
    for x in at_risk_times
]
at_risk_table.loc["HIV negative, complications so far:"] = [
    (
        ~df_cs_cox["comor_hiv"]
        & (df_cs_cox["cause"] == 2)
        & (df_cs_cox["outco_time_to_event"] <= x)
    ).sum()
    for x in at_risk_times
]
at_risk_table = at_risk_table.reset_index().rename(columns={"index": "Group"})

aj_resolved.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_resolved_onset_data___0.csv"),
    index=False)
at_risk_table.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_resolved_onset_data___1.csv"),
    index=False)

cumulative_incidence_resolved_metadata = {
    "fig_id": "tables_and_figures/fig_cumulative_incidence_resolved_onset",
    "fig_name": "fig_kaplan_meier",
    "fig_arguments": {
        "title": "Cumulative incidence, lesions resolved",
        "ylabel": "Cumulative Incidence (%)",
        "xlabel": "Time from onset (days)",
        "index_column": "Group",
        "xlim": [0, max_time],
        "base_color_map": {"HIV positive": "hsl(146,57,40)", "HIV negative": "hsl(204,61,45)"},
        "height": 700,
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": None,
        "graph_label": "Figure 5a",
        "graph_about":
            "Cumulative incidence of lesion resolution, stratified "
            "by HIV status (cause-specific, against complications). "
            "Estimated using Aalen-Johansen."
    },
    "fig_data": [
        "tables_and_figures/fig_cumulative_incidence_resolved_onset_data___0.csv",
        "tables_and_figures/fig_cumulative_incidence_resolved_onset_data___1.csv"
    ]
}

with open(os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_resolved_onset_metadata.json"), "w") as file:
    file.write(json.dumps(cumulative_incidence_resolved_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_resolved_onset_metadata.txt"), "w") as file:
    file.write(json.dumps(cumulative_incidence_resolved_metadata))

aj_compl.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_compl_onset_data___0.csv"),
    index=False)
at_risk_table.to_csv(
    os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_compl_onset_data___1.csv"),
    index=False)

cumulative_incidence_compl_metadata = {
    "fig_id": "tables_and_figures/fig_cumulative_incidence_compl_onset",
    "fig_name": "fig_kaplan_meier",
    "fig_arguments": {
        "title": "Cumulative incidence, complications",
        "ylabel": "Cumulative Incidence (%)",
        "xlabel": "Time from onset (days)",
        "index_column": "Group",
        "xlim": [0, max_time],
        "base_color_map": {"HIV positive": "hsl(19,90,50)", "HIV negative": "hsl(277,32,50)"},
        "height": 700,
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": None,
        "graph_label": "Figure 5b",
        "graph_about":
            "Cumulative incidence of complications, stratified "
            "by HIV status (cause-specific, against lesions resolved). "
            "Estimated using Aalen-Johansen."
    },
    "fig_data": [
        "tables_and_figures/fig_cumulative_incidence_compl_onset_data___0.csv",
        "tables_and_figures/fig_cumulative_incidence_compl_onset_data___1.csv"
    ]
}

with open(os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_compl_onset_metadata.json"), "w") as file:
    file.write(json.dumps(cumulative_incidence_compl_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_cumulative_incidence_compl_onset_metadata.txt"), "w") as file:
    file.write(json.dumps(cumulative_incidence_compl_metadata))

# Put together all Cox results into one CSV table


def get_table_entry(summary, covariate):
    summary_rounded = summary.round(2).astype(str)
    p = summary_rounded.loc[covariate, "p"] if summary.loc[covariate, "p"] >= 0.005 else "<0.005"
    return (
        summary_rounded.loc[covariate, "exp(coef)"]
        + " (" + summary_rounded.loc[covariate, "exp(coef) lower 95%"]
        + "-" + summary_rounded.loc[covariate, "exp(coef) upper 95%"] + '),'
        + p
    )


labels = {
    "comor_hiv": "HIV positive (vs negative)",
    "demog_sex___male": "Male (vs Female)",
    "expo14_type___sexual_transmission": "Sexual transmission (vs other transmission)",
    "D1_lesioncount": "Baseline lesion counts (/100)",
    "D1_lesion": "Basline lesion locations",
    "D1_lesion___genital": "Genital",
    "D1_lesion___limbs": "Limbs",
    "D1_lesion___face": "Face",
    "D1_lesion___torso": "Torso",
}

table = """Primary analysis: cause specific Cox models (time origin = admission), ,  ,   ,
Covariate,Lesion resolution HR (95% CI),p-value,Complications HR (95%),p-value"""

for covariate in labels.keys():
    table += "\n" + labels[covariate]
    if covariate == "D1_lesion":
        table += ",,,,"
    else:
        table += "," + get_table_entry(cph_cs_resolved.summary, covariate)
        table += "," + get_table_entry(cph_cs_compl.summary, covariate)

table += """
,,,,
Secondary analysis: cause specific Cox models (time origin = onset of symptoms),,,,
Covariate,Lesion resolution HR (95% CI),p-value,Complications HR (95%),p-value"""

for covariate in labels.keys():
    table += "\n" + labels[covariate]
    if covariate == "D1_lesion":
        table += ",,,,"
    else:
        table += "," + get_table_entry(cph_cs_onset_resolved.summary, covariate)
        table += "," + get_table_entry(cph_cs_onset_compl.summary, covariate)

table += """
,,,,
Secondary analysis: Cox model with complications by D14 and censoring on D28,,,,
Covariate,HR (95% CI),p-value,,
"""

for covariate in labels.keys():
    table += "\n" + labels[covariate]
    if covariate == "D1_lesion":
        table += ",,,,"
    else:
        table += "," + get_table_entry(cph.summary, covariate) + ",,"

with open(os.path.join(tables_and_figures_filepath, "fig_table_cox_data___0.csv"), "w") as file:
    file.write(table)


table_metadata = {
    "fig_id": "tables_and_figures/fig_table_cox",
    "fig_name": "fig_table",
    "fig_arguments": {
        "table_key": "",
        "suffix": "tables_and_figures",
        "filepath": "",
        "save_inputs": False,
        "graph_id": None,
        "graph_label": "Table 2",
        "graph_about":
            "Results from cause-specific Cox models. Each model is stratified by "
            "age group, i.e. separate baseline hazards for ages 0-4, 5-17 and 18-79."
    },
    "fig_data": ["tables_and_figures/fig_table_cox_data___0.csv"]
}

with open(os.path.join(tables_and_figures_filepath, "fig_table_cox_metadata.json"), "w") as file:
    file.write(json.dumps(table_metadata))

with open(os.path.join(tables_and_figures_filepath, "fig_table_cox_metadata.txt"), "w") as file:
    file.write(json.dumps(table_metadata))


# Could also do Fine-Gray model for competing risks (lesion resolution and complications)
# but there isn't really Python packages that can do it.

# =============================================================================
# Add other things for the VERTEX dashboard

config = {
    "project_name": "Mpox Observational Cohort Study DRC",
    "map_layout_center_latitude": -4,
    "map_layout_center_longitude": 22,
    "map_layout_zoom": 2.8
}

with open(os.path.join(OUTPUT_FILEPATH, "config_file.json"), "w") as file:
    file.write(json.dumps(config))

dashboard_metadata = {
    "insight_panels": [
        {
            "item": "Results",
            "label": "Tables and Figures",
            "suffix": "tables_and_figures",
            "graph_ids": [
                "tables_and_figures/fig_table",
                "tables_and_figures/fig_sankey",
                "tables_and_figures/fig_heatmaps",
                "tables_and_figures/fig_bar_chart",
                "tables_and_figures/fig_bar_chart_hiv_pos",
                "tables_and_figures/fig_bar_chart_hiv_neg",
                "tables_and_figures/fig_table_cox",
                "tables_and_figures/fig_cumulative_incidence_resolved",
                "tables_and_figures/fig_cumulative_incidence_compl"
                "tables_and_figures/fig_cumulative_incidence_resolved_onset",
                "tables_and_figures/fig_cumulative_incidence_compl_onset"
            ]
        }
    ]
}

with open(os.path.join(OUTPUT_FILEPATH, "dashboard_metadata.json"), "w") as file:
    file.write(json.dumps(dashboard_metadata))

pd.DataFrame(
    [["COD", "Congo, Dem. Rep.", df_map["subjid"].count()]],
    columns=["country_iso", "country_name", "country_count"]
).to_csv(os.path.join(OUTPUT_FILEPATH, "dashboard_data.csv"), index=False)

# # =============================================================================
# # Complications upset plot
#
# pyramid_columns = [
#     "comor_hiv",
#     "compl_eye",
#     "compl_neurological",
#     "compl_gynobs",
#     "compl_read"
# ]
# df_pyramid_values = get_descriptive_data(
#     df_map[pyramid_columns], dictionary, by_column=None,
#     include_sections=dictionary["section"].unique().tolist())
# df_pyramid_values["outco_binary_outcome"] = (df_map["outco_outcome"] == "Died").copy()
# add_dictionary = {
#     "field_name": ["outco_binary_outcome"],
#     "field_type": ["binary"],
#     "field_label": ["Outcome: Died"],
#     "answer_options": [""],
#     "parent": ["outco"],
#     "branching_logic": [""],
#     "section": [""],
# }
# dictionary = pd.concat(
#     [
#         dictionary,
#         pd.DataFrame.from_dict(add_dictionary, orient="index").T
#     ],
#     ignore_index=True
# )
#
# df_pyramid_values = (df_pyramid_values.groupby("comor_hiv").mean()*100).T[::-1]
#
# df_pyramid = pd.DataFrame(columns=["side", "y_axis", "stack_group", "value", "left_side"])
# df_pyramid["side"] = ["Positive"]*df_pyramid_values.shape[0]*2 + ["Negative"]*df_pyramid_values.shape[0]*2
# df_pyramid["y_axis"] = df_pyramid_values.index.tolist()*4
# severity_ind = df_pyramid["y_axis"].str.startswith("baseline_severity")
# df_pyramid["y_axis"] = df_pyramid["y_axis"].map(dict(zip(dictionary["field_name"], dictionary["field_label"])))
# df_pyramid.loc[severity_ind, "y_axis"] = "Severity at baseline: " + df_pyramid.loc[severity_ind, "y_axis"]
# df_pyramid["stack_group"] = 2*(["Yes"]*df_pyramid_values.shape[0] + ["No"]*df_pyramid_values.shape[0])
# df_pyramid["value"] = (
#     df_pyramid_values[True].tolist()
#     + (100 - df_pyramid_values[True]).tolist()
#     + df_pyramid_values[False].tolist()
#     + (100 - df_pyramid_values[False]).tolist()
# )
# df_pyramid["left_side"] = df_pyramid["side"] == "Positive"
#
# df_pyramid.to_csv(
#     os.path.join(tables_and_figures_filepath, "fig_pyramid_data___0.csv"),
#     index=False)
