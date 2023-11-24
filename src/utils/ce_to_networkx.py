import networkx as nx
from owlapy.model import (IRI, OWLClass, OWLClassExpression, OWLDataProperty,
                          OWLDataSomeValuesFrom, OWLDatatype,
                          OWLDeclarationAxiom, OWLEquivalentClassesAxiom,
                          OWLNamedIndividual, OWLObjectComplementOf,
                          OWLObjectIntersectionOf, OWLObjectMinCardinality,
                          OWLObjectProperty, OWLObjectSomeValuesFrom,
                          OWLObjectUnionOf)
from owlapy.namespaces import Namespaces
from owlapy.render import DLSyntaxObjectRenderer


def dec_tree(ce, G=None, class_node=None):
    def process_nested(ce, G):
        for _ in ce._operands:
            # process_nested(_)
            dec_tree(_, G, class_node)

    if G is None:
        G = nx.DiGraph()

    if isinstance(ce, OWLClass):
        # class_node = ce
        G.add_edge(class_node, ce._iri._remainder, label="is")

    if isinstance(ce, OWLObjectComplementOf):
        # class_node = ce
        G.add_edge(class_node, ce._operand._iri._remainder, label=" not")

    elif isinstance(ce, OWLObjectProperty):
        property_name = ce._iri.get_short_form()
        G.add_edge(class_node, property_name)

    elif isinstance(ce, OWLObjectSomeValuesFrom):
        if isinstance(ce._filler, OWLObjectIntersectionOf):
            process_nested(ce._filler, G)

        else:
            edge = ce._property._iri._remainder
            # G.add_edge(parent_node, edge)
            # dec_tree(ce._filler, G, class_node)
            G.add_edge(class_node, ce._filler._iri._remainder, label=edge)

    elif isinstance(ce, OWLObjectMinCardinality):
        min_cardinality = ce._cardinality
        property_name = ce._property._iri._remainder
        filler_name = f"not {ce._filler._operand._iri._remainder}"
        if isinstance(ce._filler, OWLObjectComplementOf):
            filler_name = f"not {filler_name}"
        G.add_edge(
            class_node,
            filler_name,
            label=f" {property_name} (min {min_cardinality} )",
        )

    elif isinstance(ce, OWLObjectIntersectionOf):
        process_nested(ce, G)

    return G


CE = OWLObjectIntersectionOf(
    (
        OWLObjectSomeValuesFrom(
            property=OWLObjectProperty(
                IRI("http://dl-learner.org/carcinogenesis#", "hasAtom")
            ),
            filler=OWLClass(IRI("http://dl-learner.org/carcinogenesis#", "Carbon-10")),
        ),
        OWLObjectMinCardinality(
            10,
            property=OWLObjectProperty(
                IRI("http://dl-learner.org/carcinogenesis#", "hasStructure")
            ),
            filler=OWLObjectComplementOf(
                OWLClass(IRI("http://dl-learner.org/carcinogenesis#", "Non_ar_6c_ring"))
            ),
        ),
    )
)
NS = "http://www.w3.org/2002/07/owl#Thing"
g1 = dec_tree(CE, class_node=NS)
print(g1)
