import json

from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.owlapy.model import IRI, OWLNamedIndividual


def train_celoe(file_path=None):
    kgs = ["mutag", "aifb"]

    for kg in kgs:
        target_dict = {}
        json_file_path = f"configs/{kg}.json"  # Replace with your JSON file path

        with open(json_file_path, "r", encoding="utf-8") as json_file:
            settings = json.load(json_file)
        target_kb = KnowledgeBase(path=settings["data_path"])
        for str_target_concept, examples in settings["problems"].items():
            positive_examples = set(examples["positive_examples"])
            negative_examples = set(examples["negative_examples"])
            print("Target concept: ", str_target_concept)

            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, positive_examples)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, negative_examples)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

            qual = Accuracy()
            heur = CELOEHeuristic(
                expansionPenaltyFactor=0.05,
                startNodeBonus=1.0,
                nodeRefinementPenalty=0.01,
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

            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]
            best_concept = hypotheses[0].concept
            concept_ind = [
                indv.get_iri().as_str()
                for indv in target_kb.individuals_set(best_concept)
            ]
            concept_length = target_kb.concept_len(hypotheses[0].concept)

            target_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
                "concept_individuals": concept_ind,
                "positive_examples": list(positive_examples),
                "negative_examples": list(negative_examples),
            }
        # Define the filename where you want to save the JSON
        file_path = f"results/predictions/CELOE/{kg}.json"

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)


def train_celoe_fid(file_path=None):
    kgs = ["mutag", "aifb"]

    for kg in kgs:
        target_dict = {}
        kg_path = f"data/KGs/{kg}.owl"
        json_file_path = (
            f"configs/{kg}_gnn_preds.json"  # Replace with your JSON file path
        )

        with open(json_file_path, "r", encoding="utf-8") as json_file:
            settings = json.load(json_file)
        target_kb = KnowledgeBase(path=kg_path)
        for str_target_concept, examples in settings.items():
            positive_examples = set(examples["positive_examples"])
            negative_examples = set(examples["negative_examples"])
            print("Target concept: ", str_target_concept)

            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, positive_examples)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, negative_examples)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

            qual = Accuracy()
            heur = CELOEHeuristic(
                expansionPenaltyFactor=0.05,
                startNodeBonus=1.0,
                nodeRefinementPenalty=0.01,
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

            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]
            best_concept = hypotheses[0].concept
            concept_ind = [
                indv.get_iri().as_str()
                for indv in target_kb.individuals_set(best_concept)
            ]
            concept_length = target_kb.concept_len(best_concept)

            target_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
                "concept_individuals": concept_ind,
                "positive_examples": list(positive_examples),
                "negative_examples": list(negative_examples),
            }
        # Define the filename where you want to save the JSON
        file_path = f"results/predictions/CELOE/{kg}_gnn_preds.json"

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
