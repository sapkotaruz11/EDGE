import os

import rdflib as rdf
from rdflib import RDF, XSD, Graph, Literal, URIRef


def pre_process_mutag():
    """
    Processes the MUTAG dataset by filtering out certain predicates and blank nodes,
    and serializes the processed graph into a new file.

    The function reads the raw MUTAG dataset, filters out triples where the predicate
    is 'isMutagenic' and either the subject or object is a blank node. It also converts
    string literals to boolean True. The processed graph is then saved in a new file.
    """
    raw_path = "data/mutag-hetero_faec5b61/mutag_stripped.nt"
    processed_path = "data/KGs/mutag_stripped_processed.nt"

    # Check if the raw dataset file exists
    if os.path.isfile(raw_path):
        # Parse the raw graph
        g_mutag = Graph().parse(raw_path)

        # Initialize a new graph for the processed data
        g_mutag_new = Graph()
        is_mutagenic = rdf.term.URIRef(
            "http://dl-learner.org/carcinogenesis#isMutagenic"
        )
        BT = Literal(True, datatype=XSD.boolean)

        # Iterate over each triple in the graph
        for s, p, o in g_mutag:
            # Skip triples with 'isMutagenic' predicate
            if p == is_mutagenic:
                continue
            # Skip triples with blank nodes
            if isinstance(s, rdf.BNode) or isinstance(o, rdf.BNode):
                continue
            # Convert string literals to boolean True
            if (
                isinstance(o, rdf.Literal)
                and str(o.datatype) == "http://www.w3.org/2001/XMLSchema#string"
            ):
                g_mutag_new.add((s, p, BT))
                continue
            # Add the triple to the new graph
            g_mutag_new.add((s, p, o))

        # Serialize the new graph to a file
        g_mutag_new.serialize(destination=processed_path, encoding="utf-8", format="nt")
    else:
        print("Raw Dataset not Available")


def pre_process_aifb():
    """
    Processes the AIFB dataset by filtering out certain predicates and blank nodes,
    and serializes the processed graph into a new file.

    The function reads the raw AIFB dataset, filters out triples where the predicate
    is 'employs' or 'affiliation' and either the subject or object is a blank node.
    It also converts string literals to boolean True. The processed graph is then
    saved in a new file.
    """
    raw_path = "data/aifb-hetero_82d021d8/aifbfixed_complete.n3"
    processed_path = "data/KGs/aifbfixed_complete_processed.n3"

    # Check if the raw dataset file exists
    if os.path.isfile(raw_path):
        # Parse the raw graph
        g_aifb = Graph().parse(raw_path)
        employs = rdf.term.URIRef("http://swrc.ontoware.org/ontology#employs")
        affiliation = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")
        BT = Literal(True, datatype=XSD.boolean)
        new_g_aifb = Graph()

        # Iterate over each triple in the graph
        for s, p, o in g_aifb:
            # Skip triples with 'employs' or 'affiliation' predicates
            if p == employs or p == affiliation:
                continue
            # Skip triples with blank nodes
            if isinstance(s, rdf.BNode) or isinstance(o, rdf.BNode):
                continue
            # Convert string literals to boolean True
            if (
                isinstance(o, rdf.Literal)
                and str(o.datatype) == "http://www.w3.org/2001/XMLSchema#string"
            ):
                new_g_aifb.add((s, p, BT))
                continue
            # Add the triple to the new graph
            new_g_aifb.add((s, p, o))

        # Serialize the new graph to a file
        new_g_aifb.serialize(destination=processed_path, encoding="utf-8", format="n3")
    else:
        print("Raw Dataset not Available")


#'Next Processing ---> Convert nt/n3 files to OWL KG Using ROBOT tool
# For AIFB remove the #Thing description from KG to make it compatible with EvoLearner as we get 'PSet Terminals have to have unique names
# As thing is already added from another instance, we can safely remove that manually to make it work
