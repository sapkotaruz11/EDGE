import json
import os
import time

from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from ontolearn.refinement_operators import ModifiedCELOERefinement
from owlapy.model import IRI, OWLNamedIndividual


def train_celoe(file_path=None, kgs=None, use_heur=False):
    if kgs is None:
        kgs = ["mutag", "aifb"]

    for kg in kgs:
        t0 = time.time()
        target_dict = {}
        json_file_path = f"configs/{kg}.json"
        # Replace with your JSON file path
        if not os.path.exists(json_file_path):
            print(
                f"JSON file not found: {json_file_path}. Please Create the learning problem for the dataset {kg}"
            )
            continue  # Skip this dataset/model combination

        with open(json_file_path, "r", encoding="utf-8") as json_file:
            settings = json.load(json_file)
        target_kb = KnowledgeBase(path=settings["data_path"])
        for str_target_concept, examples in settings["problems"].items():
            positive_examples = set(examples["positive_examples_train"])
            negative_examples = set(examples["negative_examples_train"])
            print("Target concept: ", str_target_concept)

            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, positive_examples)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, negative_examples)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

            if use_heur:
                qual = F1()
                heur = CELOEHeuristic(
                    expansionPenaltyFactor=0.05,
                    startNodeBonus=1.0,
                    nodeRefinementPenalty=0.01,
                )
                op = ModifiedCELOERefinement(
                    knowledge_base=target_kb,
                    use_negation=False,
                    use_all_constructor=False,
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
            else:
                model = CELOE(
                    knowledge_base=target_kb, quality_func=F1(), max_runtime=600
                )

            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]
            positive_examples_test = set(examples["positive_examples_test"])
            negative_examples_test = set(examples["negative_examples_test"])
            best_concept = hypotheses[0].concept
            concept_ind = set(
                [
                    indv.get_iri().as_str()
                    for indv in target_kb.individuals_set(best_concept)
                ]
            )
            concept_length = target_kb.concept_len(hypotheses[0].concept)
            concept_inds = concept_ind.intersection(
                positive_examples_test | negative_examples_test
            )

            target_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
                "concept_individuals": list(concept_inds),
                "positive_examples": list(positive_examples_test),
                "negative_examples": list(negative_examples_test),
            }
        # Define the filename where you want to save the JSON
        file_path = f"results/predictions/CELOE/{kg}.json"
        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
        t1 = time.time()
        dur = t1 - t0
        print(f"Trained CELOE  (predictions) on  {kg} dataset on {dur : .2f}")


def train_celoe_fid(file_path=None, kgs=None, use_heur=False):
    if kgs is None:
        kgs = ["mutag", "aifb"]

    for kg in kgs:
        t0 = time.time()
        target_dict = {}
        kg_path = f"data/KGs/{kg}.owl"
        json_file_path = (
            f"configs/{kg}_gnn_preds.json"  # Replace with your JSON file path
        )
        if not os.path.exists(json_file_path):
            print(
                f"JSON file not found: {json_file_path}. Please Create the learning problem for the dataset {kg}"
            )
            continue  # Skip this dataset/model combination

        with open(json_file_path, "r", encoding="utf-8") as json_file:
            settings = json.load(json_file)
        target_kb = KnowledgeBase(path=kg_path)
        for str_target_concept, examples in settings.items():
            positive_examples = set(examples["positive_examples_train"])
            negative_examples = set(examples["negative_examples_train"])
            print("Target concept: ", str_target_concept)

            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, positive_examples)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, negative_examples)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

            if use_heur:
                qual = F1()
                heur = CELOEHeuristic(
                    expansionPenaltyFactor=0.05,
                    startNodeBonus=1.0,
                    nodeRefinementPenalty=0.01,
                )
                op = ModifiedCELOERefinement(
                    knowledge_base=target_kb,
                    use_negation=False,
                    use_all_constructor=False,
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
            else:
                model = CELOE(
                    knowledge_base=target_kb, quality_func=F1(), max_runtime=600
                )

            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]
            positive_examples_test = set(examples["positive_examples_test"])
            negative_examples_test = set(examples["negative_examples_test"])
            best_concept = hypotheses[0].concept
            concept_ind = set(
                [
                    indv.get_iri().as_str()
                    for indv in target_kb.individuals_set(best_concept)
                ]
            )
            concept_length = target_kb.concept_len(hypotheses[0].concept)
            concept_inds = concept_ind.intersection(
                positive_examples_test | negative_examples_test
            )

            target_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
                "concept_individuals": list(concept_inds),
                "positive_examples": list(positive_examples_test),
                "negative_examples": list(negative_examples_test),
            }
        # Define the filename where you want to save the JSON
        file_path = f"results/predictions/CELOE/{kg}_gnn_preds.json"
        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
        t1 = time.time()
        dur = t1 - t0
        print(f"Trained CELOE  (fidelity) on  {kg} dataset on {dur : .2f}")
