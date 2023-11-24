# Code adapted from https://github.com/mathematiger/ExplwCE/blob/master/visualization.py
import colorsys
import logging
import math
import os
import re

import dgl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matplotlib")
logger.setLevel(logging.WARNING)


# ---------------- utils
def uniquify(path, extension=".pdf"):
    if path.endswith("_"):
        path += "1"
        counter = 1
    while os.path.exists(path + extension):
        counter += 1
        while path and path[-1].isdigit():
            path = path[:-1]
        path += str(counter)
    return path


def remove_integers_at_end(string):
    pattern = r"\d+$"  # Matches one or more digits at the end of the string
    result = re.sub(pattern, "", string)
    return result


def get_last_number(string):
    pattern = r"\d+$"  # Matches one or more digits at the end of the string
    match = re.search(pattern, string)
    if match:
        last_number = match.group()
        return int(last_number)
    else:
        return None


def generate_colors(num_colors):
    # Define the number of distinct hues to use
    num_hues = num_colors + 1
    # Generate a list of evenly spaced hues
    hues = [i / num_hues for i in range(num_hues)]
    # Shuffle the hues randomly
    # random.shuffle(hues)
    saturations = []
    # saturations = [0.8 for _ in range(num_colors)]
    values = []
    # values = [0.4 for _ in range(num_colors)]
    for i in range(num_colors):
        if i % 2 == 0:
            values.append(0.4)
            saturations.append(0.4)
        else:
            values.append(0.8)
            saturations.append(0.7)
    # Convert the hues, saturations, and values to RGB colors
    colors = [
        colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)
    ]
    # Convert the RGB colors to hexadecimal strings
    hex_colors = [
        f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors
    ]
    return hex_colors


def adjust_caption_size_exp(caption_length, max_size=18, min_size=8, rate=0.1):
    font_size = max_size * math.exp(-rate * caption_length)
    return max(min_size, min(font_size, max_size))


def visualize_hd(
    hd_graph,
    inverse_indices,
    addname_for_save,
    list_all_nodetypes,
    label_to_explain=None,
    add_info="",
    name_folder="",
):
    try:
        plt.clf()
    except Exception as e:
        print(f"An exception occurred while clearing the plot: {e}")
    # create data for legend and caption
    number_of_node_types_for_colors = len(list_all_nodetypes)
    colors = generate_colors(number_of_node_types_for_colors)
    if number_of_node_types_for_colors == 4:
        colors = ["#59a14f", "#f28e2b", "#4e79a7", "#e15759"]
    curent_nodetypes_to_all_nodetypes = []
    for _ in range(len(hd_graph.ndata._ntype)):
        all_nodetypes_index = list_all_nodetypes.index(hd_graph.ntypes[_])
        curent_nodetypes_to_all_nodetypes.append([_, all_nodetypes_index])
    # create nx graph to visualize
    Gnew = nx.Graph()
    homdata = dgl.to_homogeneous(hd_graph)
    homdata.edge_index = torch.stack(homdata.edges(), dim=0)
    # num_nodes_of_graph = len(homdata.ndata["_TYPE"].tolist())
    Gnew.add_nodes_from(list(range(homdata.num_nodes())))
    # add edges
    list_edges_start, list_edges_end = (
        homdata.edge_index.tolist()[0],
        homdata.edge_index.tolist()[1],
    )

    nodes_to_explain = [
        count
        for item, tensors in inverse_indices.items()
        if tensors.numel() > 0
        for count, nitem in enumerate(homdata.ndata["_TYPE"])
        if nitem.item() == hd_graph.ntypes.index(item)
        for item_count, _ in enumerate(tensors)
        if item_count in [tensor.item() for tensor in tensors]
    ]
    label_dict = {}
    node_color = []
    for count, _ in enumerate(homdata.ndata["_ID"].tolist()):
        label_dict[count] = list_all_nodetypes[homdata.ndata["_TYPE"][count].tolist()][
            :2
        ]
        if count in nodes_to_explain:
            node_color.append("#76b7b2")
        else:
            node_color.append(colors[homdata.ndata["_TYPE"][count].tolist()])

    list_edges_for_networkx = list(zip(list_edges_start, list_edges_end))

    Gnew.add_edges_from(list_edges_for_networkx)
    # color nodes
    list_node_types = list(set([tensor.item() for tensor in homdata.ndata["_TYPE"]]))
    node_labels_to_indices = dict()
    index = 0
    stop = False  # the prediction is always done for the first node
    for nodekey in list_node_types:
        if label_to_explain != None:
            if (
                str(curent_nodetypes_to_all_nodetypes[nodekey][1]) == label_to_explain
                and stop == False
            ):
                node_labels_to_indices.update({index: "*"})
                stop = True
            else:
                node_labels_to_indices.update({index: ""})
        else:
            node_labels_to_indices.update(
                {index: curent_nodetypes_to_all_nodetypes[nodekey][1]}
            )
        index += 1
    color_map_of_nodes = []
    for typeindex in list_node_types:
        color_map_of_nodes.append(
            colors[curent_nodetypes_to_all_nodetypes[typeindex][1]]
        )
    # plt
    options = {"with_labels": "True", "node_size": 500}
    nx.draw(Gnew, node_color=node_color, **options, labels=label_dict)
    # create legend
    patch_list = []
    name_list = []
    for i in range(len(hd_graph.ntypes)):
        patch_list.append(
            plt.Circle((0, 0), 0.1, fc=colors[curent_nodetypes_to_all_nodetypes[i][1]])
        )
        name_list.append(hd_graph.ntypes[i])
    # create caption
    caption_text = add_info
    caption_size = adjust_caption_size_exp(
        caption_length=len(add_info), max_size=18, min_size=8, rate=0.1
    )
    caption_position = (0.5, 0.1)
    # folder to save in:
    folder = remove_integers_at_end(addname_for_save)
    number_ce = get_last_number(addname_for_save)
    if name_folder == "":
        name_plot_save = "content/plots/" + "/" + folder + str(number_ce)
    else:
        name_plot_save = (
            name_folder + "HeteroBAShapes" + "/ce_" + str(number_ce) + "_graph_"
        )
    name_plot_save = uniquify(name_plot_save, "_wo_text.pdf")
    directory = os.path.dirname(name_plot_save)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(name_plot_save + "_wo_text.pdf", bbox_inches="tight", format="pdf")
    plt.legend(patch_list, name_list)
    plt.figtext(*caption_position, caption_text, ha="center", size=caption_size)
    name_plot_save = uniquify(name_plot_save)
    plt.savefig(name_plot_save + ".pdf", bbox_inches="tight", format="pdf")
