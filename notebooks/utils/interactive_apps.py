from PIL import Image
import dash
import torch
from dash import dcc, html, Input, Output
import plotly.express as px
import tqdm
import base64
from io import BytesIO
import numpy as np
import torchvision.transforms as T
from torchvision.utils import make_grid
from crp.image import zimage

def img_to_base64(img_array):
    pil_img = Image.fromarray(img_array)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def resize_image(img_array, size):
    pil_img = Image.fromarray(img_array)
    pil_img = pil_img.resize((size, size))  # Resize with anti-aliasing
    return np.array(pil_img)

def create_dash_app_data_perspective(samples_2d_pd, ds, percentage_images=0.1):
    SIZE_SMALL = 50
    
    app = dash.Dash(__name__)

    # Create the scatter plot using Plotly Express
    fig = px.scatter(samples_2d_pd, x="x", y="y", hover_name="id", size=[0.5]*len(samples_2d_pd), size_max=3, width=750, height=750)
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(showlegend=False, dragmode='select')  # Set the default interaction mode to box select

    # Set layout for the app
    app.layout = html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'margin': '0', 'padding': '0', 'width': '100%'}, children=[
        html.Div(style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'width': '100%'}, children=[
            # Scatter plot
            dcc.Graph(
                id='scatter-plot',
                figure=fig,
                style={'flex': '0 1 750px', 'margin': '0', 'padding': '0'},  # Fixed width for the graph
                config={'displayModeBar': True}
            ),

            # Hover image display on the right
            html.Div(id='image-display', style={'margin-left': '10px', 'flex': '0 0 auto', 'textAlign': 'center', 'display': 'flex', 'align-items': 'center'})  
        ]),

        # Selected images below the scatter plot
        html.Div(id='selected-images', style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center', 'margin-top': '10px'})  # Selected images
    ])

    
    num_images = max(1, int(len(samples_2d_pd) * percentage_images))  
    print(f"Loading initial images ({percentage_images * 100:.1f}% of {len(samples_2d_pd)}, that's {num_images}) ...")
    image_indices = np.random.choice(len(samples_2d_pd), num_images, replace=False)

    # Add images to the layout
    for index in tqdm.tqdm(image_indices):
        img_src = f"data:image/png;base64,{img_to_base64(resize_image(ds.reverse_normalization(ds[index][0]).numpy().transpose(1,2,0).astype(np.uint8), SIZE_SMALL))}"
        fig.add_layout_image(
            dict(
                source=img_src,
                x=samples_2d_pd['x'][index],
                y=samples_2d_pd['y'][index],
                xref="x",
                yref="y",
                sizex=0.05,  # Size of the image
                sizey=0.05,  # Size of the image
                xanchor="center",
                yanchor="middle",
                opacity=1,
                layer="above",
            )
        )

    # Display the image for index 0 by default
    @app.callback(
        Output('image-display', 'children'),
        Input('scatter-plot', 'hoverData'),
    )
    def display_image(hoverData):
        img_index = 0 
        if hoverData is not None:
            point_id = hoverData['points'][0]['hovertext']
            img_index = int(point_id)

        img_src = f"data:image/png;base64,{img_to_base64(ds.reverse_normalization(ds[img_index][0]).numpy().transpose(1,2,0).astype(np.uint8))}"
        return html.Div([
            html.Div(f'Index: {img_index}', style={'margin-left': '5px', 'line-height': '20px'}),
            html.Img(src=img_src, style={'width': '100px', 'height': '100px'})
        ])

    @app.callback(
        Output('selected-images', 'children'),
        Input('scatter-plot', 'selectedData')
    )
    def display_selected_images(selectedData):
        if selectedData is not None:
            points = selectedData['points']
            selected_ids = []
            
            for point in points:
                point_id = point['hovertext']
                selected_ids.append(int(point_id))

            print(f"Loading {len(selected_ids)} selected images ...")
            image_elements = []
            for img_id in tqdm.tqdm(selected_ids):
                img_src = f"data:image/png;base64,{img_to_base64(ds.reverse_normalization(ds[img_id][0]).numpy().transpose(1,2,0).astype(np.uint8))}"
                image_elements.append(
                    html.Div([
                        html.Div(f'{img_id}', style={'textAlign': 'center'}),
                        html.Img(src=img_src, style={'width': '80px', 'height': '80px', 'margin': '5px'})
                    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'})
                )

            return image_elements 

        return html.Div()
    return app

def get_concept_visualization(ref_imgs_all_concepts, cid, num_imgs=5, nrow=8):
    ref_imgs = ref_imgs_all_concepts[cid][:num_imgs]
    resize = T.Resize((150, 150))
    grid = make_grid([resize(torch.from_numpy(np.copy(np.asarray(img))).permute((2, 0, 1))) for img in ref_imgs],
                     padding=2, nrow=nrow)
    grid = np.array(zimage.imgify(grid.detach().cpu())).astype(np.uint8)
    return grid


def create_dash_app_model_perspective(concept_2d_pd, ref_imgs_all_concepts, percentage_images=0.1):
    
    app = dash.Dash(__name__)

    # Create the scatter plot using Plotly Express
    fig = px.scatter(concept_2d_pd, x="x", y="y", hover_name="id", size=[0.5]*len(concept_2d_pd), size_max=3, width=750, height=750)
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(showlegend=False, dragmode='select')  # Set the default interaction mode to box select

    # Set layout for the app
    app.layout = html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'margin': '0', 'padding': '0', 'width': '100%'}, children=[
        html.Div(style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'width': '100%'}, children=[
            # Scatter plot
            dcc.Graph(
                id='scatter-plot',
                figure=fig,
                style={'flex': '0 1 750px', 'margin': '0', 'padding': '0'},  # Fixed width for the graph
                config={'displayModeBar': True}
            ),

            # Hover image display on the right
            html.Div(id='image-display', style={'margin-left': '10px', 'flex': '0 0 auto', 'textAlign': 'center', 'display': 'flex', 'align-items': 'center'})  
        ]),

        # Selected images below the scatter plot
        html.Div(id='selected-images', style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center', 'margin-top': '10px'})  # Selected images
    ])

    
    num_images = max(1, int(len(concept_2d_pd) * percentage_images))  
    print(f"Loading initial concept visualizations ({percentage_images * 100:.1f}% of {len(concept_2d_pd)}, that's {num_images}) ...")
    image_indices = np.random.choice(len(concept_2d_pd), num_images, replace=False)

    # Add images to the layout
    for index in tqdm.tqdm(image_indices):
        img_src = f"data:image/png;base64,{img_to_base64(get_concept_visualization(ref_imgs_all_concepts, index, num_imgs=4, nrow=2))}"
        fig.add_layout_image(
            dict(
                source=img_src,
                x=concept_2d_pd['x'][index],
                y=concept_2d_pd['y'][index],
                xref="x",
                yref="y",
                sizex=0.04,  # Size of the image
                sizey=0.04,  # Size of the image
                xanchor="center",
                yanchor="middle",
                opacity=1,
                layer="above",
            )
        )

    # Display the image for index 0 by default
    @app.callback(
        Output('image-display', 'children'),
        Input('scatter-plot', 'hoverData'),
    )
    def display_image(hoverData):
        concept_id = 0 
        if hoverData is not None:
            point_id = hoverData['points'][0]['hovertext']
            concept_id = int(point_id)

        img_src = f"data:image/png;base64,{img_to_base64(get_concept_visualization(ref_imgs_all_concepts, concept_id))}"
        return html.Div([
            html.Div(f'Index: {concept_id}', style={'margin-left': '5px', 'line-height': '20px'}),
            html.Img(src=img_src, style={'width': '300px', 'height': '60px'})
        ])

    @app.callback(
        Output('selected-images', 'children'),
        Input('scatter-plot', 'selectedData')
    )
    def display_selected_images(selectedData):
        if selectedData is not None:
            points = selectedData['points']
            selected_ids = []
            
            for point in points:
                point_id = point['hovertext']
                selected_ids.append(int(point_id))

            print(f"Loading {len(selected_ids)} selected concepts {selected_ids} ...")
            image_elements = []
            for concept_id in tqdm.tqdm(selected_ids):
                img_src = f"data:image/png;base64,{img_to_base64(get_concept_visualization(ref_imgs_all_concepts, concept_id, num_imgs=16))}"
                image_elements.append(
                    html.Div([
                        html.Div(f'Neuron #{concept_id}', style={'textAlign': 'center'}),
                        html.Img(src=img_src, style={'width': '600px', 
#                                                      'height': '100px', 
                                                     'margin': '5px'})
                    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'})
                )

            return image_elements 

        return html.Div()
    return app