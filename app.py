import math
import os
import json

import streamlit as st
import yaml
from dotenv import load_dotenv

import pandas as pd
import igraph
from igraph import Graph
from igraph import EdgeSeq
import plotly.graph_objects as go

import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter

load_dotenv()

# Load from the JSON file
with open("subject_color_map.json", "r") as f:
    subject_color_map = json.load(f)

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]


def make_curriculum_graph(parent_node, edges=[], labels=[], annotations=[]):
    parent_id = parent_node["id"]

    for child in parent_node["children"]:
        child_id = child["id"]
        edges.append((parent_id, child_id))
        labels.append("Node ID: {}<br>{}".format(child["id"], child["label"]))
        annotations.append(child["annotations"])
        
        if len(child["children"]) > 0:
            edges, labels, annotations = make_curriculum_graph(child, edges=edges, labels=labels, annotations=annotations)

    return edges, labels, annotations

def make_igraph_tree_plot_data(section_edges, annotations, queried_node_list=None):
    # TODO: not a very readable function, but this should be reimplmented in frontend in the future
    def make_annotations(pos, annotations, font_size=10, font_color='black'):
        L=len(pos)
        if len(annotations)!=L:
            raise ValueError('The lists pos and text must have the same len')
        hover_text_dicts = []
        for k in range(L):
            hover_text_dicts.append(
                dict(
                    text=annotations[k], 
                    x=pos[k][0], y=2*M-position[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return hover_text_dicts
    
    def make_node_opacities(pos, queried_node_list):
        L = len(pos)

        opacities = [0.25 for _ in range(L)]
        node_colors = ["white" for _ in range(L)]

        for node in queried_node_list:
            opacities[int(node["node_id"])] = float(node["certainty"])
            node_colors[int(node["node_id"])] = "orange"

        return opacities, node_colors

    G = Graph.TupleList(section_edges, directed=True)
    nr_vertices = G.vcount()
    lay = G.layout('rt', root=[0])
    position = {k: (lay[k][0], lay[k][1] + math.sin(lay[k][0])*0.0) for k in range(nr_vertices)}

    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)
    es = EdgeSeq(G) # sequence of edges
    E = [e.tuple for e in G.es] # list of edges
    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    igraph_annotations = make_annotations(position, annotations)
    node_opacities = 0.8
    node_colros = "white"

    if queried_node_list is not None:
        node_opacities, node_colors = make_node_opacities(position, queried_node_list)

    return Xe, Ye, Xn, Yn, position, igraph_annotations, node_opacities, node_colors



def get_curriculum_tree_graph_object(curriculum_tree, queried_node_list=None):
    # TODO: the initial labels could be added to make_curriculum_graph() function
    labels = []
    annotations = []

    labels.append(curriculum_tree["label"])
    annotations.append(curriculum_tree["annotations"])

    section_edges, v_labels, annotations = make_curriculum_graph(curriculum_tree, edges=[], labels=labels, annotations=annotations)

    Xe, Ye, Xn, Yn, position, igraph_annotations, node_opacities, node_colors = make_igraph_tree_plot_data(section_edges, annotations, queried_node_list=queried_node_list)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                        y=Ye,
                        mode='lines',
                        line=dict(color='rgb(210,210,210)', width=1),
                        hoverinfo='none'
                        ))
    fig.add_trace(go.Scatter(x=Xn,
                        y=Yn,
                        mode='markers',
                        name='bla',
                        marker=dict(symbol='circle-dot',
                                        size=10,
                                        # color="white",   
                                        color=node_colors,
                                        line=dict(color='black', width=0.5), 
                                        opacity=node_opacities
                                        ),
                        # opacity=node_opacities,
                        text=v_labels,
                        hoverinfo='text',
                        ))

    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )

    fig.update_layout(title='Japanese Jr. High School Social Study Curriculum as a Tree',
                    annotations=igraph_annotations,
                    font_size=12,
                    showlegend=False,
                    xaxis=axis,
                    yaxis=axis,
                    margin=dict(l=40, r=40, b=85, t=100),
                    hovermode='closest',
                    plot_bgcolor='white',
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=7.5,
                        font_family="Rockwell",
                        namelength=10,  # This ensures that the hover text is not truncated
                        align='left',  # Align text to the left
                        # width=100  # Set a fixed width for the hover text
                    ),
                    height=800  # Set the height of the figure
                    )
    return fig


def query_vectors(query_text, search_limit, selected_subjects=None):
    class_name = "JapaneseCurriculumDemo"

    # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
        auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
        headers={"X-OpenAI-Api-Key": openai_api_key},           # Replace with your Cohere API key
    )

    collection = client.collections.get(class_name)

    # object_filter_list = [Filter.by_property("subject").equal(subject) for subject in selected_subjects]

    response = collection.query.near_text(
        query=query_text,
        limit=search_limit, 
        return_metadata=MetadataQuery(distance=True, certainty=True), 
        # filters=(
        #     Filter.any_of(object_filter_list)
        # )
    )

    client.close()  # Free up resources

    queried_node_list = []
    
    for obj in response.objects:
        row_dict = {}
        row_dict["node_id"] = obj.properties["node_id"]
        row_dict["label"] = obj.properties["label"]
        row_dict["subject"] = obj.properties["subject"]
        # row_dict["paragraph_idx"] = int(obj.properties["paragraph_idx"])
        row_dict["certainty"] = obj.metadata.certainty
        row_dict["distance"] = obj.metadata.distance
        
        queried_node_list.append(row_dict)
    
    return queried_node_list


# Read YAML file
with open("tree_data/sample_structured_curriculum.yaml", "r", encoding="utf-8") as file:
    curriculum_tree = yaml.safe_load(file)  # Load YAML content as a dictionary

st.set_page_config(layout="wide")

query_text = st.text_input("Write a query to search contents of curriculum:", "コンプライアンス")

text_search_limit = st.slider("Select a number of texts to query", 
                              min_value=0, 
                              max_value=100, 
                              value=100, 
                              step=1
                              )

if query_text :
    queried_node_list = query_vectors(query_text, text_search_limit)

    fig = get_curriculum_tree_graph_object(curriculum_tree, queried_node_list=queried_node_list)

st.plotly_chart(fig)