import json

from ontolearn.concept_learner import EvoLearner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.owlapy.model import IRI, OWLNamedIndividual


def train_evo(file_path=None, kgs=["aifb"]):
    if kgs is None:
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

            model = EvoLearner(knowledge_base=target_kb, max_runtime=600)
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
        file_path = f"results/predictions/EVO/{kg}.json"

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)


def train_evo_fid(file_path=None):
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

            model = EvoLearner(knowledge_base=target_kb, max_runtime=600)
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
        file_path = f"results/predictions/EVO/{kg}_gnn_preds.json"

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
