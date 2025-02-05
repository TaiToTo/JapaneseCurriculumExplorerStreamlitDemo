import streamlit as st
import yaml
import igraph
from igraph import Graph
from igraph import EdgeSeq
import math
import plotly.graph_objects as go

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

def make_igraph_tree_plot_data(section_edges, annotations):
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

    G = Graph.TupleList(section_edges, directed=True)
    nr_vertices = G.vcount()
    lay = G.layout('rt', root=[0])
    position = {k: (lay[k][0], lay[k][1] + math.sin(lay[k][0])*0.1) for k in range(nr_vertices)}

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

    return Xe, Ye, Xn, Yn, position, igraph_annotations

st.set_page_config(layout="wide")


# Read YAML file
with open("tree_data/sample_structured_curriculum.yaml", "r", encoding="utf-8") as file:
    curriculum_tree = yaml.safe_load(file)  # Load YAML content as a dictionary

# TODO: the initial labels could be added to make_curriculum_graph() function
labels = []
annotations = []

labels.append(curriculum_tree["label"])
annotations.append(curriculum_tree["annotations"])

section_edges, v_labels, annotations = make_curriculum_graph(curriculum_tree, edges=[], labels=labels, annotations=annotations)

Xe, Ye, Xn, Yn, position, igraph_annotations = make_igraph_tree_plot_data(section_edges, annotations)

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
                                    color="white",    #'#DB4551',
                                    line=dict(color='black', width=0.5)
                                    ),
                    text=v_labels,
                    hoverinfo='text',
                    opacity=0.8
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

st.plotly_chart(fig)