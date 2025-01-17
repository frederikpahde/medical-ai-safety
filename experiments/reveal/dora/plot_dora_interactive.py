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
from crp.image import zimage
from crp.visualization import FeatureVisualization
from dash import Dash, Input, Output, dcc, html
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from zennit.composites import EpsilonPlusFlat

from datasets import load_dataset
from models import get_canonizer, get_fn_model_loader
from utils.dimensionality_reduction import get_2d_data
from utils.helper import load_config
from utils.render import vis_opaque_img_border

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--aggr', type=str, default="avg")
    parser.add_argument('--emb', type=str, default="umap")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--class_id', type=int, default=1)
    parser.add_argument('--ref_type', type=str, default="real_rel")
    parser.add_argument('--config_file',
                        default="config_files/revealing/hyper_kvasir_attacked/local/resnet50d_identity_2.yaml")

    return parser

app = Dash(__name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([

    html.Div(children=[
        html.H1('DORA UMAP'),
        dbc.Button(id='button', n_clicks=0, children='Run', color="secondary", outline=True,
                style={"display": "none"}),
        dcc.Loading(children=[
            dcc.Graph(
                id='umap',
                responsive=True,
                config={'scrollZoom': True, 'displaylogo': False},
                style={"width": "100%", "height": "75vh"})
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
        sizex=2,
        sizey=2,
        sizing="stretch",
        xanchor="center",
        yanchor="middle",
    )
    return fig['layout']['images'][0]


@app.callback(Output('umap', 'figure'), Input('button', 'n_clicks'), )
def main(n_clicks):
    
    print("in main")

    fname_distances = f"{config['dir_precomputed_data']}/dora_data/{config['dataset_name']}_{config['model_name']}_{config['aggr']}/distances/{config['layer_name']}_{config['ref_type']}_{class_id}.pth"
    D = torch.load(fname_distances)
    print(f"Loading existing distances from {fname_distances}")

    X = get_2d_data(D, config["emb"], metric="precomputed")
    max_diff = max(X[:, 0].max() - X[:, 0].min(), X[:, 1].max() - X[:, 1].min())
    desired_diff = 50
    X = X * desired_diff / max_diff

    x, y = X[:, 0], X[:, 1]
    fig = px.scatter(x=x, y=y, title=None, hover_data={'id': np.arange(len(x))})
    print("Computing reference images...")
    print(f"Shape: {X.shape}, {x.shape}")
    
    if len(x) > 2000:
        Nth = 32
    elif len(x) > 1000:
        Nth = 16
    else:
        Nth = 4

    ref_imgs = fv.get_max_reference(range(len(x))[::Nth], config['layer_name'], "relevance", (0, 1), rf=True,
                                    composite=composite, plot_fn=vis_opaque_img_border)

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
    print("Returning Fig")

    return fig

@app.callback(
    Output('image_container', 'children'),
    Input('umap', 'selectedData'))

def display_img(data):
    if data is None:
        return {}
    children = []
    for k, point in enumerate(data["points"]):
        concept_id = point["pointIndex"]
        print("load images for", concept_id)
        ref_imgs = fv.get_max_reference([concept_id], config['layer_name'],
                                        "relevance", r_range=(0, 16), composite=composite, rf=True,
                                        plot_fn=vis_opaque_img_border)[concept_id]

        n = 5
        sams_imgs = collect_sams_imgs(concept_id, n)
        imgs = [np.array(x.resize((100, 100))) for x in ref_imgs]

        fig = px.imshow(np.array(imgs), facet_col=0, binary_string=True, labels={},
                        facet_col_spacing=0, facet_row_spacing=0.2, facet_col_wrap=8)

        ## sample-IDs
        sample_ids = d_c_sorted[:16, concept_id]
        targets = [fv.dataset.get_target(i) for i in sample_ids]
        titles = [f"Label: {t}" for t in targets]

        for i, title in enumerate(titles):
            fig.layout.annotations[i]['text'] = title

        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(hoverinfo='none', hovertemplate=None)

        fig_sams = px.imshow(np.array(sams_imgs), facet_col=0, binary_string=True, labels={},
                        facet_col_spacing=0, facet_row_spacing=0, facet_col_wrap=8)

        fig_sams.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
        fig_sams.update_xaxes(showticklabels=False)
        fig_sams.update_yaxes(showticklabels=False)
        fig_sams.update_traces(hoverinfo='none', hovertemplate=None)
        fig.update_layout(margin=dict(t=20, b=0, l=0, r=0))
        child = dcc.Graph(
            responsive=True,
            figure=fig,
            id="images",
            style={"width": "100%", 
                   "height": "250px", 
                   "padding": "0", 
                   "margin": "1em auto", 
                   "max-width": "1000px"},
        )

        child_sams = dcc.Graph(
            responsive=True,
            figure=fig_sams,
            id="images_sams",
            style={"width": "100%", "height": "300px", "padding": "0", "margin": "1em auto", "max-width": "1000px"},
        )

        title = html.H3(f"Concept {concept_id}", style={"margin": "1em auto", "max-width": "1000px"})
        subtitle_relmax = html.H4(f"RelMax", style={"margin": "1em auto", "max-width": "1000px"})
        subtitle_sams = html.H4(f"sAMS", style={"margin": "1em auto", "max-width": "1000px"})
        modal = html.Div(
            [
                dbc.Button("Collect Reference Samples", id=f"open_{k}", n_clicks=0, key=concept_id,
                           name=concept_id),
            ]
        )

        children.append(title)
        children.append(modal)
        children.append(subtitle_relmax)
        children.append(child)
        children.append(subtitle_sams)
        children.append(child_sams)

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


def collect_sams_imgs(concept, n):
    resize = Resize((100,100))
    imgs = [resize(torch.from_numpy(np.asarray(Image.open(f"{sams_dir}/{concept}_{c}+.jpg"))).permute((2, 0, 1))).numpy().transpose(1,2,0) for c in range(n)]
    return imgs

def show_relmax_refimgs(ref_imgs, axs):
    resize = Resize((150, 150))
    for r, (concept, imgs) in enumerate(ref_imgs.items()):
        ax = axs[r]
        grid = make_grid(
        [resize(torch.from_numpy(np.asarray(img)).permute((2, 0, 1))) for img in imgs],
            padding=2)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        ax.imshow(grid)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(f"Concept {concept}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    config["aggr"] = args.aggr
    config["emb"] = args.emb
    config["ref_type"] = args.ref_type
    class_id = args.class_id
    ## prepare sAMS stuff
    sams_dir = f"{config['dir_precomputed_data']}/dora_data/{config['dataset_name']}_{config['model_name']}_{config['aggr']}/sAMS/{config['config_name']}/"

    ref_split = args.split
    ref_dataset = load_dataset(config, normalize_data=False)
    splits = {
        "train": ref_dataset.idxs_train,
        "val": ref_dataset.idxs_val,
        "test": ref_dataset.idxs_test,
        }
    ref_dataset = ref_dataset if (ref_split is None) or (ref_split=="all") else ref_dataset.get_subset_by_idxs(splits[ref_split])
    model = get_fn_model_loader(model_name=config["model_name"])(n_class=len(ref_dataset.classes),
                                                                ckpt_path=config['ckpt_path']).to("cuda").eval()

    canonizers = get_canonizer(config["model_name"])
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()

    layer_names = [config['layer_name']]
    layer_map = {layer: cc for layer in layer_names}

    attribution = CondAttribution(model)

    cache = ImageCache()
    fv_name = f"{config['dir_precomputed_data']}/crp_files/{config['dataset_name']}_{ref_split}_{config['model_name']}"

    if class_id is not None:
        fv_name = f"{fv_name}_class{class_id}"
        idxs_class = [i for i in range(len(ref_dataset)) if ref_dataset.get_target(i) == class_id]
        ref_dataset = ref_dataset.get_subset_by_idxs(idxs_class)

    fv = FeatureVisualization(attribution, ref_dataset, layer_map, preprocess_fn=ref_dataset.normalize_fn,
                            path=fv_name, cache=None)

    d_c_sorted, _, _ = load_maximization(fv.RelMax.PATH, config["layer_name"])
  
    app.run_server(debug=True, dev_tools_hot_reload=False, port=8051)
