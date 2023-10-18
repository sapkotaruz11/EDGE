from ontolearn.concept_learner import CELOE
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.model import IRI
from owlapy.model import IRI
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import IRI, OWLNamedIndividual
from ontolearn.metrics import Accuracy
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.refinement_operators import ModifiedCELOERefinement

import json


def train_celoe(file_path=None):
    kgs = ["mutag", "lymphography"]

    for kg in kgs:
        json_file_path = f"configs/{kg}.json"  # Replace with your JSON file path

        # Open and read the JSON file line by line
        with open(json_file_path, "r") as json_file:
            settings = json.load(json_file)

        target_kb = KnowledgeBase(path=settings["data_path"])
        str_target_concept = settings["target_concept"]
        pos_examples = set(settings["positive_examples"])
        neg_examples = set(settings["negative_examples"])

        print("Target concept: ", str_target_concept)

        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, pos_examples)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, neg_examples)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

        qual = Accuracy()
        heur = CELOEHeuristic(
            expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01
        )
        op = ModifiedCELOERefinement(
            knowledge_base=target_kb, use_negation=False, use_all_constructor=False
        )

        model = CELOE(
            knowledge_base=target_kb,
            max_runtime=600,
            refinement_operator=op,
            quality_func=qual,
            heuristic_func=heur,
            max_num_of_concepts_tested=10_000_000_000,
            iter_bound=10_000_000_000,
        )

        model.fit(lp)

        model.save_best_hypothesis(
            n=3, path="results/actuals/CELOE/Predictions_{0}".format(str_target_concept)
        )
        # Get Top n hypotheses
        hypotheses = list(model.best_hypotheses(n=3))
        # Use hypotheses as binary function to label individuals.
        # predictions = model.predict(individuals=list(typed_pos | typed_neg),
        #                               hypotheses=hypotheses)
        # print(predictions)
        [print(_) for _ in hypotheses]
        best_concept = hypotheses[0].concept
        concept_ind = [str(indv) for indv in target_kb.individuals_set(best_concept)]
        concept_length = target_kb.concept_len(hypotheses[0].concept)

        target_dict = {
            "concept_name": str_target_concept,
            "concept_length": concept_length,
            "concept_individuals": concept_ind,
        }
        # Define the filename where you want to save the JSON
        file_path = f"results/actuals/CELOE/{kg}.json"

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
