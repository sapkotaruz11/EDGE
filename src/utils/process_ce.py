"""This script was not utilized as a part of the EDGE framework. The conversion of CEs to NetworkX still is a future work."""
import networkx as nx
from owlapy.model import (IRI, OWLClass, OWLClassExpression, OWLDataProperty,
                          OWLDataPropertyDomainAxiom, OWLDataSomeValuesFrom,
                          OWLDatatype, OWLDeclarationAxiom,
                          OWLEquivalentClassesAxiom, OWLLiteral,
                          OWLNamedIndividual, OWLObjectComplementOf,
                          OWLObjectIntersectionOf, OWLObjectMinCardinality,
                          OWLObjectProperty, OWLObjectSomeValuesFrom,
                          OWLObjectUnionOf, _OWLLiteralImplInteger)
from owlapy.namespaces import Namespaces
from owlapy.render import DLSyntaxObjectRenderer


def dec_tree(ce, G=None, class_node=None):
    # Set default value for class_node if not provided
    if class_node is None:
        class_node = "http://www.w3.org/2002/07/owl#Thing"

    # Function to process nested class expressions
    def process_nested(ce, G):
        for _ in ce._operands:
            # Recursive call to process each operand
            dec_tree(_, G, class_node)

    # Initialize a directed graph if not provided
    if G is None:
        G = nx.DiGraph()

    # Check if the class expression is an OWL class
    if isinstance(ce, OWLClass):
        # Add an edge to the graph representing the class
        G.add_edge(class_node, ce._iri._remainder, label="is")

    # Check if the class expression is an object complement
    elif isinstance(ce, OWLObjectComplementOf):
        # Add an edge with label 'not'
        G.add_edge(class_node, ce._operand._iri._remainder, label=" not")

    # Check if the class expression is an object property
    elif isinstance(ce, OWLObjectProperty):
        # Extract the property name and add an edge
        property_name = ce._iri.get_short_form()
        G.add_edge(class_node, property_name)

    # Check if the class expression is an object with some values
    elif isinstance(ce, OWLObjectSomeValuesFrom):
        # Process nested intersection if filler is an intersection
        if isinstance(ce._filler, OWLObjectIntersectionOf):
            process_nested(ce._filler, G)
        else:
            # Add an edge with the property as the label
            edge = ce._property._iri._remainder
            G.add_edge(class_node, ce._filler._iri._remainder, label=edge)

    # Check if the class expression is an object with minimum cardinality
    elif isinstance(ce, OWLObjectMinCardinality):
        min_cardinality = ce._cardinality
        property_name = ce._property._iri._remainder
        try:
            filler_name = ce._filler._operand._iri._remainder
        except:
            filler_name = ce._filler._iri._remainder
        # Handle complement case
        if isinstance(ce._filler, OWLObjectComplementOf):
            filler_name = f"not {filler_name}"
        # Add an edge with minimum cardinality in the label
        G.add_edge(
            class_node,
            filler_name,
            label=f" {property_name} (min {min_cardinality} )",
        )

    # Check if the class expression is an intersection
    elif isinstance(ce, OWLObjectIntersectionOf):
        # Process nested expressions
        process_nested(ce, G)

    # Return the constructed graph
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
