import glob
import json
import os
import torch
from argparse import ArgumentParser

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px
import numpy as np
from experiments.sample_retrieval.run_biased_sample_ranking import run_cav_artifact_ranking
from utils.helper import load_config
from datasets import load_dataset
import torchvision.transforms as T

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--config_file', 
                        default="config_files/revealing/isic/local/resnet50d_identity_2.yaml", 
                        type=str)
    parser.add_argument('--artifacts_file', 
                        default="data/artifact_samples/artifact_samples_isic.json", 
                        type=str)
    parser.add_argument('--artifact', 
                        default="band_aid", 
                        type=str)
    parser.add_argument('--min_score', default=-16, type=float)
    parser.add_argument('--skip_known_artifacts', default=True, type=bool)
    return parser

app = dash.Dash(__name__, 
                suppress_callback_exceptions=True, 
                external_stylesheets=[dbc.themes.BOOTSTRAP])

resize = T.Resize((100,100))
app.layout = html.Div([
    html.H1('Top-N Images by CAV Score'),
    dcc.Dropdown(id='top-n-dropdown', options=[{'label': i, 'value': i} for i in [10,25,50,100,200]], value=100),
    html.Div(id='page-controls', children=[
        dbc.Button("Previous", id='prev-button', n_clicks=0, className="ml-2"),
        dbc.Button("Next", id='next-button', n_clicks=0, className="ml-2"),
        html.Div(id='page-info', style={"margin": "10px 0"}),
    ]),
    html.Div(id='image-container'),
])

@app.callback(
    Output('image-container', 'children'),
    Output('page-info', 'children'),
    Input('top-n-dropdown', 'value'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    State('top-n-dropdown', 'value'),
    State('prev-button', 'n_clicks'),
    State('next-button', 'n_clicks')
)
def update_images(top_n, prev_clicks, next_clicks, top_n_state, prev_clicks_state, next_clicks_state):
    current_page = max(0, next_clicks_state - prev_clicks_state)
    start_index = current_page * top_n_state
    end_index = start_index + top_n_state
    
    sorted_indices = sorted(range(len(cav_scores)), key=lambda i: cav_scores[i], reverse=True)
    cav_scores_sorted = list(np.array(cav_scores)[sorted_indices])
    if skip_known_artifacts:
        cav_scores_sorted = [s for s, i in zip(cav_scores_sorted, sorted_indices) if i not in dataset.sample_ids_by_artifact[artifact_name_orig]]
        sorted_indices = [i for i in sorted_indices if i not in dataset.sample_ids_by_artifact[artifact_name_orig]]
    
    
    if min_score is not None:
        keep = [s >= min_score for s in cav_scores_sorted]
        cav_scores_sorted = [s for s, k in zip(cav_scores_sorted, keep) if k]
        sorted_indices = [i for i, k in zip(sorted_indices, keep) if k]

    paged_indices = sorted_indices[start_index:end_index]
    
    checklist_values = []
    children = []
    for idx in paged_indices:
        image = dataset[idx][0]
        image_resized = resize(image)
        image_np = (image_resized * 255).type(torch.uint8).numpy().transpose((1, 2, 0))
        fig = px.imshow(image_np, labels={}, height=100)
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(hoverinfo='none', hovertemplate=None)

        for annot in fig.layout.annotations:
            annot['text'] = ''

        child = (dcc.Graph(
            responsive=True,
            figure=fig,
            style={"width": "125px", "height": "125px", "padding": "0", "margin": "0em 0"},
            config={
                'displayModeBar': False
            }
        ))
        children.append(child)
        checklist_values.append(dataset.get_sample_name(idx))
    
    checklist = dcc.Checklist(
        [
            {
                "label": [
                    child
                ],
                "value": checklist_values[i]
            }
            for i, child in enumerate(children)
        ],
        id="checklist",
        inline=True,
        value=checklist_values,
        labelStyle={"display": "flex", "align-items": "center", "justify-content": "center",
                    "flex-direction": "column", "row-gap": ".2em"},
        style={"display": "flex", "flex-wrap": "wrap", "width": "100%",
                "justify-content": "space-around", "row-gap": "1em"}
    )
    
    json_files = glob.glob("data/*.json")
    default_file = artifacts_file if artifacts_file in json_files else json_files[0]
    default_value = artifact_name_orig

    combined_layout = html.Div([
       dcc.Loading(dbc.ModalBody(
                children=[html.H3("Please choose the reference images that correspond to an artifact."),
                          html.Div(children=checklist)])),
        dbc.ModalFooter(
                [
                    html.Div([dbc.Label("Select artifact JSON file:"),
                              dbc.Select(
                                  json_files,
                                  default_file,
                                  id="json-select",
                              ), ]),
                    html.Div([
                        html.Div([dbc.Label("Artifact Name:"),
                                  dbc.Input(value=default_value, type="text", id="artifact_name"),
                                  dbc.RadioItems(
                                      options=[
                                          {"label": "Replace artifact samples", "value": 1},
                                          {"label": "Add artifact samples", "value": 2},
                                      ],
                                      value=2,
                                      id="switches-input",
                                      style={"margin": "1em 0"}
                                  ),
                                  dbc.Button("Select/Deselect All", id="select_all", className="ml-2", n_clicks=0),  
                                  dbc.Button("Save to JSON", id=f"export", className="ml-2", n_clicks=0),

                                  ], ),

                    ]),
                    dbc.Alert(
                        "Saved successfully!",
                        id="alert-auto",
                        color="success",
                        is_open=False,
                        duration=3000,
                    ),
                    dbc.Button(
                        "Close", id=f"close", className="ms-auto", n_clicks=0
                    ),

                ],
                style={"align-items": "flex-start", }
            ),
    ])
    
    total_pages = (len(cav_scores_sorted) + top_n_state - 1) // top_n_state
    page_info_text = f"Page {current_page + 1} of {total_pages}"
    return combined_layout, page_info_text

@app.callback(
    Output("checklist", "value"),
    Input("select_all", "n_clicks"),
    State("checklist", "options"),
    State("checklist", "value")
)
def toggle_select_all(n_clicks, options, selected_values):
    all_values = [option["value"] for option in options]
    if n_clicks % 2 == 1:
        # Select all
        return all_values
    else:
        # Deselect all
        return []
    
@app.callback(
    [Output("alert-auto", "is_open"), Output("alert-auto", "color"), Output("alert-auto", "children"),
     Output("export", "n_clicks")],
    Input("export", "n_clicks"),
    State("json-select", "value"),
    State("artifact_name", "value"),
    State("switches-input", "value"),
    State("checklist", "value"),
)
def export_json(n_clicks, json_file, artifact_name, switches_input, checklist):
    if n_clicks:
        print("export json")
        print(json_file, artifact_name, switches_input)

        if not artifact_name:
            return [True, "warning", "Artifact name required!", n_clicks]

        if not checklist:
            return [True, "warning", "No samples chosen!", n_clicks]

        with open(json_file, "r") as f:
            data = json.load(f)
        data.setdefault(artifact_name, [])
        if switches_input == 1:
            data[artifact_name] = []
        for i in checklist:
            if i not in data[artifact_name]:
                data[artifact_name].append(i)
        data[artifact_name] = sorted(data[artifact_name])
        with open(json_file, "w") as f:
            json.dump(data, f, indent=1)
        return [True, "success", "Saved successfully!", n_clicks]

    return [False, "", "", n_clicks]

if __name__ == "__main__":
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    split = "all"
    dataset = load_dataset(config, normalize_data=False)
    artifact_name_orig = args.artifact
    min_score = args.min_score
    skip_known_artifacts = args.skip_known_artifacts
    artifacts_file = args.artifacts_file
    fraction = 1
    cav_scores_file = f"{config['dir_precomputed_data']}/cav_scores/{config['dataset_name']}_{split}_{config['model_name']}/{config['layer_name']}_{artifact_name_orig}_{fraction}.pth"
    if not os.path.isfile(cav_scores_file):
        print("Computing CAV Scores")
        config["artifacts_file"] = args.artifacts_file
        batch_size = 16
        run_cav_artifact_ranking(config, args.artifact, fraction, batch_size, 'plot_files/data_annotation/')
    
    cav_scores = torch.load(cav_scores_file).tolist()  
    app.run_server(debug=True, dev_tools_hot_reload=False, port=8051)