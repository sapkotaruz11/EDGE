import os

import rdflib as rdf
from rdflib import RDF, XSD, Graph, Literal, URIRef


def pre_process_mutag():
    raw_path = "data/mutag/mutag-hetero_faec5b61/mutag_stripped.nt"
    processed_path = "data/mutag/mutag-hetero_faec5b61/mutag_stripped_processed.nt"

    if os.path.isfile(raw_path):
        g_mutag = Graph().parse(path=raw_path)
        g_mutag_new = Graph()
        is_mutagenic = rdf.term.URIRef(
            "http://dl-learner.org/carcinogenesis#isMutagenic"
        )
        BT = Literal(True, datatype=XSD.boolean)
        for s, p, o in g_mutag:
            if p == is_mutagenic:
                continue
            if isinstance(s, rdf.BNode) or isinstance(o, rdf.BNode):
                continue
            if isinstance(o, rdf.Literal):
                if str(o.datatype) == "http://www.w3.org/2001/XMLSchema#string":
                    g_mutag_new.add((s, p, BT))
                    continue
            g_mutag_new.add((s, p, o))
        g_mutag_new.serialize(destination=processed_path, encoding="utf-8")
    else:
        print("Raw Dataset not Available")


def pre_process_aifb():
    raw_path = "data/aifb/aifb-hetero_82d021d8/aifbfixed_complete.n3"
    processed_path = "data/aifb/aifb-hetero_82d021d8/aifbfixed_complete_processed.n3"

    if os.path.isfile(raw_path):
        g_aifb = Graph().parse(path=raw_path)
        employs = rdf.term.URIRef("http://swrc.ontoware.org/ontology#employs")
        affiliation = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")
        BT = Literal(True, datatype=XSD.boolean)
        new_g_aifb = Graph()

        for s, p, o in g_aifb:
            if p == employs or p == affiliation:
                continue
            if isinstance(s, rdf.BNode) or isinstance(o, rdf.BNode):
                continue
            if isinstance(o, rdf.Literal):
                if str(o.datatype) == "http://www.w3.org/2001/XMLSchema#string":
                    new_g_aifb.add((s, p, BT))
                    continue
            new_g_aifb.add((s, p, o))
        new_g_aifb.serialize(destination=processed_path, encoding="utf-8")
    else:
        print("Raw Dataset not Available")


#'Next Processing ---> Convert nt/n3 files to OWL KG Using ROBOT tool
# For AIFB remove the #Thing description from KG to make it compatible with EvoLearner as we get 'PSet Terminals have to have unique names
# As thing is already added from another instance, we can safely remove that manually to make it work
