import os
from itertools import product
import ast
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, ALL, ctx, MATCH, dash_table
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Format, Scheme
import dash
from collections import defaultdict
from pdf2image import convert_from_path
from io import BytesIO
import base64
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

# ---------- CONFIG ----------
BASE_DIR = "figures"
PDF_SUFFIXES = ["main_choice_h0", "main_choice_None", "cumulative_fr_0", "cumulative_fr_h0", "firing_rates"]
METRIC_GROUPS = [
    "EMD_%s_STIM", "EMD_%s_UNIFORM", "%s_CORRECT_BIAS_MSE", "%s_INCORRECT_BIAS_MSE",
    "OBLIQUE_%s_SKEW", "OBLIQUE_%s_SKEW_H0", "NEAR_OBLIQUE_%s_SKEW", "NEAR_OBLIQUE_%s_SKEW_H0",
    "CARDINAL_%s_SKEW", "CARDINAL_%s_SKEW_H0", "NEAR_CARDINAL_%s_SKEW", "NEAR_CARDINAL_%s_SKEW_H0",
    "OBLIQUE_%s_N_PEAKS", "NEAR_OBLIQUE_%s_N_PEAKS", "CARDINAL_%s_N_PEAKS", "NEAR_CARDINAL_%s_N_PEAKS",
    "CENTER_%s_N_PEAKS", "NEAR_CENTER_%s_N_PEAKS"
]
EXPECTED_PDFS = 5


# ---------- Parameter Parsing ----------
def get_model_results():
    dfs = []
    for filename in os.listdir('results'):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join('results', filename))
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop(columns=["Unnamed: 0"])
    return df
    # return pd.read_csv("model_res.csv").drop(columns=["Unnamed: 0"])


def parse_real_folder_structure(base_path):
    param_tree = defaultdict(set)

    for root, dirs, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)
        path_parts = rel_path.split(os.sep)

        if rel_path == ".":
            continue

        last_name = ""
        for depth, name in enumerate(path_parts):
            try:
                param_name, param_val = name.split("-", 1)
            except ValueError:
                last_name = name
                continue
            try:
                param_val = float(param_val)
            except ValueError:
                continue
            if last_name:
                param_name = f"{last_name}_{param_name}"
                last_name = ""
            param_tree[param_name].add(param_val)

    filtered_params = {
        name: sorted(vals) for name, vals in param_tree.items() if len(vals) > 1
    }
    return filtered_params


def pdf_to_base64_image(pdf_path, dpi=150):
    try:
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
        if not images:
            return None
        buffer = BytesIO()
        images[0].save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
        return None


def get_leaf_folders(base_path):
    leaf_folders = []
    for root, dirs, files in os.walk(base_path):
        if len([f for f in files if f.endswith(".pdf")]) >= EXPECTED_PDFS:
            leaf_folders.append(os.path.relpath(root, base_path))
    return leaf_folders


def parse_folder_to_param_values(folder_path):
    parts = folder_path.split(os.sep)
    values = {}
    last_name = ""
    for part in parts:
        try:
            name, val = part.split("-", 1)
        except ValueError:
            last_name = part
            continue
        try:
            val = float(val)
        except ValueError:
            continue
        if last_name:
            name = f"{last_name}_{name}"
            last_name = ""
        values[name] = val
    return values


# ---------- Start Dash App ----------
app = Dash(__name__)
server = app.server

params = parse_real_folder_structure(BASE_DIR)
leaf_folders = get_leaf_folders(BASE_DIR)
param_names = list(params.keys())
df = get_model_results()
app.layout = html.Div([
    html.H2("Simulation PDF Explorer"),
    html.Label("Select Breakdown Parameter:"),
    dcc.Dropdown(
        id="breakdown-param",
        options=[{"label": p, "value": p} for p in param_names],
        value=param_names[0]
    ),
    html.Div(id="fixed-selectors"),
    # add dash_table that is filtered by the selected fixed parameters
    html.Div(id="parameter-table-container"),
    html.Div(id="multi-metric-graphs"),

    html.Button("Load PDFs", id="load-btn", n_clicks=0),
    html.Br(),
    html.Label("Select Figure Type:"),
    dcc.Dropdown(
        id="figure-type-selector",
        options=[{"label": f, "value": f} for f in PDF_SUFFIXES],
        value=PDF_SUFFIXES[0],
        disabled=True
    ),
    html.Div(id="pdf-display")
])


@app.callback(
    Output("multi-metric-graphs", "children"),
    Input("breakdown-param", "value"),
    Input({"type": "fixed", "index": ALL}, "value"),
    State({"type": "fixed", "index": ALL}, "id")
)
def update_multiple_graphs(breakdown_param, fixed_vals, fixed_ids):
    if not breakdown_param or not fixed_vals or not fixed_ids:
        raise PreventUpdate

    fixed_map = {id_["index"]: val for id_, val in zip(fixed_ids, fixed_vals)}

    filtered_df = df.copy()
    for param, val in fixed_map.items():
        if param in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[param] == val]

    if breakdown_param not in filtered_df.columns:
        raise PreventUpdate

    filtered_df = filtered_df.sort_values(by=breakdown_param)

    import plotly.graph_objects as go
    graph_figures = []

    for metric in METRIC_GROUPS:
        fig = go.Figure()

        for prefix in ["NDR", "IDR"]:
            col = metric % prefix
            if col in filtered_df.columns:
                fig.add_trace(go.Scatter(
                    x=filtered_df[breakdown_param],
                    y=filtered_df[col],
                    mode="lines+markers",
                    name=col
                ))

        fig.update_layout(
            title=(metric % prefix).replace("_", " "),
            xaxis_title=breakdown_param,
            yaxis_title="Value",
            template="plotly_white",
            height=300
        )

        graph_figures.append(fig)
    # do the same graph for the maximal peak strength for the IDR and NDR models
    for prefix in ["NDR", "IDR"]:
        for loc in ["OBLIQUE", "NEAR_OBLIQUE", "CARDINAL", "NEAR_CARDINAL", "CENTER", "NEAR_CENTER"]:
            col = f"{loc}_{prefix}_PEAK_STRENGTH"
            if col in filtered_df.columns:
                fig = go.Figure()
                # peak_strengths is an array of peak strengths. You should calculate the max peak strength for each location,
                # and plot it over the breakdown parameter
                peak_strengths = filtered_df[col]
                if peak_strengths.isnull().all():
                    continue
                try:
                    fig.add_trace(go.Scatter(
                        x=filtered_df[breakdown_param],
                        y=peak_strengths.apply(lambda x: max(ast.literal_eval(x))),
                        mode="lines+markers",
                        name=col
                    ))
                except:
                    continue
                fig.update_layout(
                    title=f"{loc} {prefix} Max Peak Strength",
                    xaxis_title=breakdown_param,
                    yaxis_title="Max Peak Strength",
                    template="plotly_white",
                    height=300
                )
                graph_figures.append(fig)

    return html.Div(
        children=[
            html.Div(
                dcc.Graph(figure=fig, style={"height": "300px", "width": "100%"}),
                style={"padding": "10px"}
            )
            for fig in graph_figures
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "10px"
        }
    )


@app.callback(
    Output("parameter-table-container", "children"),
    Input({"type": "fixed", "index": ALL}, "value"),
    State({"type": "fixed", "index": ALL}, "id")
)
def update_parameter_table(fixed_vals, fixed_ids):
    if not fixed_vals or not fixed_ids:
        raise PreventUpdate

    fixed_map = {id_["index"]: val for id_, val in zip(fixed_ids, fixed_vals)}

    filtered_df = df.copy()
    for param, val in fixed_map.items():
        if param in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[param] == val]

    return html.Div([
        html.H4("Filtered Parameter Table"),
        dash_table.DataTable(
            id='parameter-table',
            columns=[{
                "name": col,
                "id": col,
                "type": "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "text",
                "format": Format(precision=2) if pd.api.types.is_numeric_dtype(df[col]) else None
            } for col in filtered_df.columns],
            data=filtered_df.to_dict('records'),
            filter_action="native",
            sort_action="native",
            page_action="native",
            page_size=20,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        )
    ])


@app.callback(
    Output("fixed-selectors", "children"),
    Input("breakdown-param", "value")
)
def update_fixed_param_selectors(breakdown_param):
    return [
        html.Div([
            html.Label(f"Fix value for {p}:"),
            dcc.Dropdown(
                id={"type": "fixed", "index": p},
                options=[{"label": str(v), "value": v} for v in params[p]],
                value=params[p][0]
            )
        ]) for p in param_names if p != breakdown_param
    ]


@app.callback(
    Output("figure-type-selector", "disabled"),
    Output("pdf-display", "children", allow_duplicate=True),
    Input("load-btn", "n_clicks"),
    State("figure-type-selector", "value"),
    State("breakdown-param", "value"),
    State({"type": "fixed", "index": ALL}, "value"),
    State({"type": "fixed", "index": ALL}, "id"),
    prevent_initial_call=True
)
def load_pdfs(n_clicks, selected_type, breakdown_param, fixed_vals, fixed_ids):
    if n_clicks == 0 or not fixed_vals or not fixed_ids:
        raise PreventUpdate

    fixed_map = {id_["index"]: val for id_, val in zip(fixed_ids, fixed_vals)}

    matched_folders = []
    folder_map = {}

    for folder in leaf_folders:
        param_vals = parse_folder_to_param_values(folder)
        if all(param_vals.get(k) == v for k, v in fixed_map.items()) and breakdown_param in param_vals:
            b_val = param_vals[breakdown_param]
            matched_folders.append(b_val)
            folder_map[b_val] = folder

    matched_folders = sorted(matched_folders)

    if not matched_folders:
        return True, [html.P("No matching folders found.")]

    display_store = {}
    for val in matched_folders:
        path = os.path.join(BASE_DIR, folder_map[val])
        files = [f for f in os.listdir(path) if f.endswith(".pdf")]
        display_store[val] = {
            f: os.path.join(path, f) for f in files
        }
    print(display_store)
    app.display_cache = display_store
    content = _show_images(breakdown_param, selected_type)
    # call the update_figures_display function to update the display
    # update_figures_display(selected_type, breakdown_param)
    return False, content  # Enable dropdown, clear display


@app.callback(
    Output("pdf-display", "children", allow_duplicate=True),
    Input("figure-type-selector", "value"),
    State("breakdown-param", "value"),
    prevent_initial_call=True
)
def update_figures_display(selected_type, breakdown_param):
    if not hasattr(app, "display_cache") or selected_type is None:
        print('error')
        raise PreventUpdate

    content = _show_images(breakdown_param, selected_type)

    return content


def _show_images(breakdown_param, selected_type):
    content = []
    row = []
    for val in sorted(app.display_cache.keys()):
        file_path = app.display_cache[val].get(selected_type + ".pdf")
        if not file_path:
            print(f"File not found for {val} with type {selected_type}: {file_path}, options: {app.display_cache[val]}")
            continue
        img = pdf_to_base64_image(file_path)
        if img:

            col = html.Div([
                html.H5(f"{breakdown_param} = {val}"),
                html.Img(src=img, style={"width": "100%", "height": "auto", "margin-bottom": "20px"})
            ], style={"width": "48%", "display": "inline-block", "padding": "1%"})
            row.append(col)
            if len(row) == 2:
                content.extend(row)
                row = []
    if row:
        content.extend(row)
    return content


if __name__ == "__main__":
    app.run(debug=True, port=8051)
