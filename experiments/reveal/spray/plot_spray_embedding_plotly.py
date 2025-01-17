import glob
import json
import multiprocessing
import random
from argparse import ArgumentParser

import dash_bootstrap_components as dbc
import h5py
import numpy as np
import plotly.express as px
import torch
from PIL import Image
from dash import Dash, html, dcc, Output, Input, State
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets import load_dataset
from utils.helper import load_config

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([

    html.Div(children=[
        html.H1('SpRAy Explanation Embedding'),
        dbc.Button(id='button', n_clicks=0, children='Run', color="secondary", outline=True,
                   style={"display": "none"}),
        dcc.Loading(children=[
            dcc.Graph(
                id='t-sne',
                responsive=True,
                config={'scrollZoom': True, 'displaylogo': False},
                style={"width": "100%", "height": "75vh"})
        ], type="circle"),
    ]),
    html.H1('Chosen Input Images'),
    dcc.Loading(children=[
        html.Div(id="image_container"),
    ], type="circle"),
], style={"max-width": "1000px", "margin": "0 auto"})


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--class_id', type=int, default=4)
    parser.add_argument('--emb_type', type=str, default="tsne", choices=["umap", "tsne"])
    parser.add_argument('--config_file',
                        default="config_files/revealing/isic/local/resnet50d_identity_2.yaml")

    return parser

def add_plot(fig, x_, y_, img):
    fig.add_layout_image(
        x=x_,
        y=y_,
        source=img,
        xref="x",
        yref="y",
        sizex=4,
        sizey=4,
        sizing="stretch",
        xanchor="center",
        yanchor="middle",
    )
    return fig['layout']['images'][0]


@app.callback(Output('t-sne', 'figure'), Input('button', 'n_clicks'), )
def main(n_clicks):
    with h5py.File(analysis_file, 'r') as fp:
        X = fp[str(class_idx)]['embedding'][emb_type][::1]
    x, y = X[:, 0], X[:, 1]
    fig = px.scatter(x=x, y=y, title=None, hover_data={'id': sample_ids})
    print("Computing reference images...")
    print(f"Plotting... ({len(x)})")

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    nth = 10 if len(x) > 1000 else 8
    res = [pool.apply_async(add_plot, args=(fig, x_, y_, img)) for x_, y_, img in zip(x[::nth], y[::nth], imgs[::nth])]
    for r in res:
        r.wait()

    fig_imgs = []
    for r in res:
        fig_imgs.append(r.get())

    pool.close()
    pool.join()
    fig['layout']['images'] = [f for f in fig_imgs]

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    fig.update_layout(dragmode='select', autosize=False)
    return fig


@app.callback(
    Output('image_container', 'children'),
    Input('t-sne', 'selectedData'))
def display_img(data):
    if data is None:
        return {}
    children = []

    imgs_choice = [imgs[x["pointIndex"]] for x in data["points"]]
    for img in imgs_choice:
        fig = px.imshow(img, labels={}, height=100)
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

    checklist = dcc.Checklist(
        [
            {
                "label": [
                    child
                ],
                "value": sample_ids[data["points"][i]["pointIndex"]]
            }
            for i, child in enumerate(children)
        ],
        id="checklist",
        inline=True,
        value=[sample_ids[data["points"][i]["pointIndex"]] for i in range(len(children))],
        labelStyle={"display": "flex", "align-items": "center", "justify-content": "center",
                    "flex-direction": "column", "row-gap": ".2em"},
        style={"display": "flex", "flex-wrap": "wrap", "width": "100%",
               "justify-content": "space-around", "row-gap": "1em"}
    )

    json_files = glob.glob("data/artifact_samples/*.json")
    footer = html.Div(
        [
            html.Div([dbc.Label("Select artifact JSON file:"),
                      dbc.Select(
                          json_files,
                          json_files[0],
                          id="json-select",
                      ), ]),
            html.Div([
                html.Div([dbc.Label("Artifact Name:"),
                          dbc.Input(placeholder="Name goes here...", type="text", id="artifact_name"),
                          dbc.RadioItems(
                              options=[
                                  {"label": "Replace artifact samples", "value": 1},
                                  {"label": "Add artifact samples", "value": 2},
                              ],
                              value=2,
                              id="switches-input",
                              style={"margin": "1em 0"}
                          ),
                          dbc.Button("Save to JSON", id=f"export", className="ml-2", n_clicks=0),

                          ], ),
            ]),
            dbc.Alert(
                "Saved successfully!",
                id="alert-auto",
                color="success",
                is_open=False,
                duration=3000,
            )
        ],
        style={"align-items": "flex-start", "display": "flex", "margin-top": "2em", "column-gap": "1em"})

    return [html.Div(children=checklist), footer]


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
        with open(json_file, "w") as f:
            json.dump(data, f, indent=1)
        return [True, "success", "Saved successfully!", n_clicks]
    return [False, "", "", n_clicks]


if __name__ == "__main__":
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    class_idx = args.class_id
    emb_type = args.emb_type
    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    split="all"
    dataset = load_dataset(config, normalize_data=False)
    splits = {
            "train": dataset.idxs_train,
            "val": dataset.idxs_val,
            "test": dataset.idxs_test,
            }
    dataset = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])

    results_dir = config['dir_precomputed_data']
    analysis_file = f"{results_dir}/spray/{config['dataset_name']}/{config['config_name']}_{split}.hdf5"

    with h5py.File(analysis_file, 'r') as fp:
        sample_ids = np.array(fp[str(class_idx)]['index'])

    sample_ids = np.array(sample_ids)

    dataset = dataset.get_subset_by_idxs(sample_ids)

    # make sure to store file names for ISIC
    if "isic" in dataset_name:
        sample_ids = dataset.metadata.image.values

    dataset.transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    print("Collect Images..")

    dl = DataLoader(dataset=dataset, num_workers=multiprocessing.cpu_count() - 1, batch_size=256)
    imgs = []

    for d, t in dl:
        imgs.extend([Image.fromarray((x.numpy() * 255).astype(np.uint8)) for x in d.permute((0, 2, 3, 1))])

    app.run_server(debug=True, dev_tools_hot_reload=False, port=8051)