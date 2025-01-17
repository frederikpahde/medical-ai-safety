import glob
import json
import multiprocessing
import random
from argparse import ArgumentParser
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import torch
from crp.attribution import CondAttribution
from crp.cache import ImageCache
from crp.concepts import ChannelConcept
from crp.helper import load_maximization
from crp.image import vis_opaque_img
from crp.visualization import FeatureVisualization
from dash import Dash, html, dcc, Output, Input, State, callback_context
from zennit.composites import EpsilonPlusFlat
from datasets import load_dataset
from models import get_fn_model_loader, get_canonizer
from utils.dimensionality_reduction import get_2d_data
from utils.helper import load_config

from utils.render import vis_opaque_img_border

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--class_id', default=0, type=int)
    parser.add_argument('--sim_key', default="rel_cosine_sim_max", type=str)
    parser.add_argument('--embedding', default="umap", type=str)
    parser.add_argument('--ignore_unused_concepts', default=True, type=bool)
    parser.add_argument('--config_file',
                        default="config_files/revealing/hyper_kvasir_attacked/local/resnet50d_identity_2.yaml")

    return parser

app = Dash(__name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([

    html.Div(children=[
        html.H1('Concept Embedding'),
        dbc.Button(id='button', n_clicks=0, children='Run', color="secondary", outline=True,
                style={"display": "none"}),
        dcc.Loading(children=[
            dcc.Graph(
                id='t-sne',
                responsive=True,
                config={'scrollZoom': True, 'displaylogo': False},
                style={"width": "100%", "height": "100vh"})
        ], type="circle"),
    ]),
    html.H1('Concept Visualizations'),
    dcc.Loading(children=[
        html.Div(id="image_container"),
    ], type="circle"),
], style={"max-width": "1000px", "margin": "0 auto"})

def add_plot(fig, x_, y_, img):
    fig.add_layout_image(
        x=x_,
        y=y_,
        source=img,
        xref="x",
        yref="y",
        sizex=.3,
        sizey=.3,
        sizing="stretch",
        xanchor="center",
        yanchor="middle",
    )
    return fig['layout']['images'][0]


@app.callback(Output('t-sne', 'figure'), Input('button', 'n_clicks'), )
def main(n_clicks):
    
    print("in main")

    try:
        data = torch.load(saving_path)
        print(f"Loading similarities from {saving_path}")
    except:
        raise ValueError()

    layer_name = config['layer_name']
    cos_pos = data[sim_key]

    X = (1-cos_pos).clamp(min=0)
    if ignore_unused_concepts:
        ignore_idxs = data["ignore_idxs"]
        D_used = X[~ignore_idxs][:,~ignore_idxs]
        X_used = get_2d_data(D_used, algorithm=embedding_algorithm, metric="precomputed")
        X = np.zeros((len(ignore_idxs), 2))
        X[~ignore_idxs] = X_used
    else:
        X = get_2d_data(X, algorithm=embedding_algorithm, metric="precomputed")

    x, y, ids = X[:, 0], X[:, 1], np.arange(len(X))
    if ignore_unused_concepts:
        x = x[~ignore_idxs]
        y = y[~ignore_idxs]
        ids = ids[~ignore_idxs]
        
    fig = px.scatter(x=x, y=y, title=None, hover_data={'id': ids})
    print("Computing reference images...")
    print(f"Shape: {X.shape}, {x.shape}")
    
    if len(x) > 2000:
        Nth = 64
    elif len(x) > 1000:
        Nth = 8 
    else:
        Nth = 16
    
    ref_imgs = fv.get_max_reference(ids[::Nth], layer_name, "relevance", (0, 1), rf=rf,
                                    composite=composite, plot_fn=vis_opaque_img_border, batch_size=BATCH_SIZE)

    imgs = [x[0] for x in ref_imgs.values()]

    print("Plotting...")

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    res = [pool.apply_async(add_plot, args=(fig, x_, y_, img)) for x_, y_, img in zip(x[::Nth], y[::Nth], imgs)]

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
    print("in display_img")
    if data is None:
        return {}
    children = []
    for k, point in enumerate(data["points"]):
        concept_id = point["customdata"][0]
        print("load images for", concept_id)
        ref_imgs = fv.get_max_reference([concept_id], config['layer_name'],
                                        "relevance", r_range=(0, 16), composite=composite, rf=rf,
                                        plot_fn=vis_opaque_img_border, batch_size=BATCH_SIZE)[concept_id]

        imgs = [np.array(x.resize((100, 100))) for x in ref_imgs]

        fig = px.imshow(np.array(imgs), facet_col=0, binary_string=True, labels={},
                        facet_col_spacing=0, facet_row_spacing=0, facet_col_wrap=8)

        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(hoverinfo='none', hovertemplate=None)

        for annot in fig.layout.annotations:
            annot['text'] = ''

        child = dcc.Graph(
            responsive=True,
            figure=fig,
            id="images",
            style={"width": "100%", "height": "200px", "padding": "0", "margin": "1em auto", "max-width": "1000px"},
        )
        title = html.H3(f"Concept {concept_id}", style={"margin": "1em auto", "max-width": "1000px"})
        modal = html.Div(
            [
                dbc.Button("Collect Reference Samples", id=f"open_{k}", n_clicks=0, key=concept_id,
                           name=concept_id),
            ]
        )

        children.append(title)
        children.append(modal)
        children.append(child)

    for l in range(k + 1, 20):
        children.append(
            dbc.Button("Collect Reference Samples", id=f"open_{l}", n_clicks=0, key=l, style={"display": "none"}))
       
    children.append(dcc.Loading(dbc.Modal(
        children=[
            dbc.ModalHeader(dbc.ModalTitle("Header")),
            dcc.Loading(dbc.ModalBody("This is the content of the modal")),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id=f"close", className="ms-auto", n_clicks=0
                )
            ),
        ],
        id="modal",
        is_open=False,
        size="xl",
    )), )
    return children

@app.callback(
    Output("modal", "is_open"),
    *[[Input(f"open_{concept_id}", "n_clicks"), Input(f"open_{concept_id}", "key")] for concept_id in
      range(20)],
    Input(f"close", "n_clicks"),
    State("modal", "is_open"),
)
def toggle_modal(*args):
    print("in toggle modeal")
    trigger = callback_context.triggered[0]
    print("You clicked button {}".format(trigger["prop_id"]))
    is_open = args[-1]
    n2 = args[-2]
    if trigger["prop_id"] != "." and "close" not in trigger["prop_id"]:

        n1 = args[int(trigger["prop_id"].split(".")[0].split("_")[-1]) * 2]
        key = args[int(trigger["prop_id"].split(".")[0].split("_")[-1]) * 2 + 1]
        if n1 or n2:
            return not is_open
    if n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal", "children"),
    *[[Input(f"open_{concept_id}", "n_clicks"), Input(f"open_{concept_id}", "key")] for concept_id in
      range(20)],
    Input("modal", "is_open"),
)
def change_modal(*args_):
    trigger = callback_context.triggered[0]
    print("You clicked button {}".format(trigger["prop_id"]))
    if trigger["prop_id"] != "." and args_[-1]:
        key = args_[int(trigger["prop_id"].split(".")[0].split("_")[-1]) * 2 + 1]

        concept_id = key
        print("load images for", concept_id)
        ref_imgs = fv.get_max_reference([concept_id], config['layer_name'],
                                        "relevance", r_range=(0, 80), composite=composite, rf=rf,
                                        plot_fn=vis_opaque_img,  batch_size=BATCH_SIZE)[concept_id]
        ref_sample_ids = load_maximization(fv.RelMax.PATH, config['layer_name'])[0][:, concept_id]
        children = []
        size=32
        imgs = [np.array(x.resize((size, size))) for x in ref_imgs]
        for img in np.array(imgs[:]):
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
                    "value": dataset.get_sample_name(ref_sample_ids[i])
                }
                for i, child in enumerate(children)
            ],
            id="checklist",
            inline=True,
            value=[dataset.get_sample_name(ref_sample_ids[i]) for i in range(len(children))],
            labelStyle={"display": "flex", "align-items": "center", "justify-content": "center",
                        "flex-direction": "column", "row-gap": ".2em"},
            style={"display": "flex", "flex-wrap": "wrap", "width": "100%",
                   "justify-content": "space-around", "row-gap": "1em"}
        )

        json_files = glob.glob("data//artifact_samples/*.json")

        return [
            dbc.ModalHeader(dbc.ModalTitle(f"Collect Reference Samples for Concept {key}")),
            dcc.Loading(dbc.ModalBody(
                children=[html.H3("Please choose the reference images that correspond to an artifact."),
                          html.Div(children=checklist)])),
            dbc.ModalFooter(
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
            ), ]
    else:
        return dcc.Loading([
            dbc.ModalHeader(dbc.ModalTitle("")),
            dbc.ModalBody("Loading..."),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id=f"close", className="ms-auto", n_clicks=0
                )
            ),
        ])

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
        with open(json_file, "w") as f:
            json.dump(data, f, indent=1)
        return [True, "success", "Saved successfully!", n_clicks]

    return [False, "", "", n_clicks]


if __name__ == "__main__":
    args = get_parser().parse_args()
    sim_key = args.sim_key
    embedding_algorithm = args.embedding
    ignore_unused_concepts = args.ignore_unused_concepts
    config = load_config(args.config_file)

    dataset = load_dataset(config, normalize_data=False)

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }

    split = "train"
    dataset = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])

    print(f"Shape dataset: {len(dataset)}")
    fv_name = f"{config['dir_precomputed_data']}/crp_files/{config['dataset_name']}_{split}_{config['model_name']}"
    class_id = args.class_id


    model = get_fn_model_loader(model_name=config["model_name"])(n_class=len(dataset.classes),
                                                                ckpt_path=config['ckpt_path']).to("cuda").eval()

    canonizers = get_canonizer(config["model_name"])
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()

    layer_names = [config['layer_name']]
    layer_map = {layer: cc for layer in layer_names}

    attribution = CondAttribution(model)
    rf = "vit" not in config["model_name"]

    cache = ImageCache()
    BATCH_SIZE=64
    fv = FeatureVisualization(attribution, dataset, layer_map, path=fv_name,
                            preprocess_fn=dataset.normalize_fn) 

    saving_path = f"{config['dir_precomputed_data']}/concept_clustering/{config['dataset_name']}_{split}_{config['model_name']}/similarities/{config['layer_name']}_{class_id}.pth"
        
    app.run_server(debug=True, dev_tools_hot_reload=False, port=8051)
