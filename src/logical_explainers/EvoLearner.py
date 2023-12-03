import json
import os
import time

EPSILON = 1e-10  # Small constant to avoid division by zero
from ontolearn.concept_learner import EvoLearner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from owlapy.model import IRI, OWLNamedIndividual


def train_evo(file_path=None, kgs=None):
    if kgs is None:
        kgs = ["mutag", "aifb"]

    for kg in kgs:
        t0 = time.time()
        target_dict = {}
        json_file_path = f"configs/{kg}.json"  # Replace with your JSON file path

        with open(json_file_path, "r", encoding="utf-8") as json_file:
            settings = json.load(json_file)
        target_kb = KnowledgeBase(path=settings["data_path"])
        for str_target_concept, examples in settings["problems"].items():
            positive_examples = set(examples["positive_example"])
            negative_examples = set(examples["negative_examples"])
            print("Target concept: ", str_target_concept)

            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, positive_examples)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, negative_examples)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

            model = EvoLearner(
                knowledge_base=target_kb, max_runtime=600, quality_func=F1()
            )
            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]

            predictions = model.predict(
                individuals=list(typed_pos | typed_neg), hypotheses=hypotheses
            )
            best_concept = hypotheses[0].concept
            # concept_ind = set(
            #     [
            #         indv.get_iri().as_str()
            #         for indv in target_kb.individuals_set(best_concept)
            #     ]
            # )
            concept_length = target_kb.concept_len(hypotheses[0].concept)
            # concept_ind = concept_ind.intersection(
            #     positive_examples | negative_examples
            # )
            pos_preds = list(
                predictions[predictions.columns[0]][
                    predictions[predictions.columns[0]] == 1.0
                ].index
            )
            if kg == "mutag":
                pos = [item.split("#")[1] for item in positive_examples]
                neg = [item.split("#")[1] for item in negative_examples]
            if kg == "aifb":
                pos = [item.split("/")[-1] for item in positive_examples]
                neg = [item.split("/")[-1] for item in negative_examples]

            target_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
                "concept_individuals": pos_preds,
                "positive_examples": pos,
                "negative_examples": neg,
            }
            # print(target_dict)

        # Define the filename where you want to save the JSON
        file_path = f"results/predictions/EVO/{kg}.json"
        # Check if the file exists
        if os.path.exists(file_path):
            # Remove the file
            os.remove(file_path)

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
        t1 = time.time()
        dur = t1 - t0
        print(f"Trained EvoLearner (prediction) on {kg} dataset on {dur : .2f}")


def train_evo_fid(file_path=None, kgs=None):
    if kgs is None:
        kgs = ["mutag", "aifb"]

    for kg in kgs:
        t0 = time.time()
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

            model = EvoLearner(
                knowledge_base=target_kb, max_runtime=600, quality_func=F1()
            )
            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]
            predictions = model.predict(
                individuals=list(typed_pos | typed_neg), hypotheses=hypotheses
            )
            best_concept = hypotheses[0].concept
            # concept_ind = set(
            #     [
            #         indv.get_iri().as_str()
            #         for indv in target_kb.individuals_set(best_concept)
            #     ]
            # )
            concept_length = target_kb.concept_len(hypotheses[0].concept)
            # concept_ind = concept_ind.intersection(
            #     positive_examples | negative_examples
            # )
            pos_preds = list(
                predictions[predictions.columns[0]][
                    predictions[predictions.columns[0]] == 1.0
                ].index
            )
            if kg == "mutag":
                pos = [item.split("#")[1] for item in positive_examples]
                neg = [item.split("#")[1] for item in negative_examples]
            if kg == "aifb":
                pos = [item.split("/")[-1] for item in positive_examples]
                neg = [item.split("/")[-1] for item in negative_examples]

            target_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
                "concept_individuals": pos_preds,
                "positive_examples": pos,
                "negative_examples": neg,
            }

        # Define the filename where you want to save the JSON
        file_path = f"results/predictions/EVO/{kg}_gnn_preds.json"
        # Check if the file exists
        if os.path.exists(file_path):
            # Remove the file
            os.remove(file_path)

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
        t1 = time.time()
        dur = t1 - t0
        print(f"Trained EvoLearner (fidelity) on {kg} dataset on {dur : .2f}")


def train_evo_train_test(file_path=None, kgs=None):
    if kgs is None:
        kgs = ["mutag", "aifb"]

    for kg in kgs:
        t0 = time.time()
        target_dict = {}
        json_file_path = (
            f"configs/{kg}_train_test.json"  # Replace with your JSON file path
        )

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

            model = EvoLearner(
                knowledge_base=target_kb, max_runtime=600, quality_func=F1()
            )
            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]
            positive_examples_test = set(examples["positive_examples_test"])
            negative_examples_test = set(examples["negative_examples_test"])
            typed_pos_test = set(
                map(OWLNamedIndividual, map(IRI.create, positive_examples_test))
            )
            typed_neg_test = set(
                map(OWLNamedIndividual, map(IRI.create, negative_examples_test))
            )
            predictions = model.predict(
                individuals=list(typed_pos_test | typed_neg_test), hypotheses=hypotheses
            )
            best_concept = hypotheses[0].concept
            # concept_ind = set(
            #     [
            #         indv.get_iri().as_str()
            #         for indv in target_kb.individuals_set(best_concept)
            #     ]
            # )
            concept_length = target_kb.concept_len(hypotheses[0].concept)
            # concept_ind = concept_ind.intersection(
            #     positive_examples | negative_examples
            # )
            pos_preds = list(
                predictions[predictions.columns[0]][
                    predictions[predictions.columns[0]] == 1.0
                ].index
            )
            if kg == "mutag":
                pos = [item.split("#")[1] for item in positive_examples_test]
                neg = [item.split("#")[1] for item in negative_examples_test]
            if kg == "aifb":
                pos = [item.split("/")[-1] for item in positive_examples_test]
                neg = [item.split("/")[-1] for item in negative_examples_test]

            target_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
                "concept_individuals": pos_preds,
                "positive_examples": pos,
                "negative_examples": neg,
            }
            # print(target_dict)
        # Define the filename where you want to save the JSON
        file_path = f"results/predictions/EVO/{kg}_train_test.json"
        # Check if the file exists
        if os.path.exists(file_path):
            # Remove the file
            os.remove(file_path)

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
        t1 = time.time()
        dur = t1 - t0
        print(
            f"Trained EvoLearner Train-test-split on (prediction) {kg} dataset on {dur : .2f}"
        )


def train_evo_train_test_fid(file_path=None, kgs=None):
    if kgs is None:
        kgs = ["mutag", "aifb"]

    for kg in kgs:
        t0 = time.time()
        kg_path = f"data/KGs/{kg}.owl"
        target_dict = {}
        json_file_path = f"configs/{kg}_gnn_preds_train_test.json"  # Replace with your JSON file path

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

            model = EvoLearner(
                knowledge_base=target_kb, max_runtime=600, quality_func=F1()
            )
            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]
            positive_examples_test = set(examples["positive_examples_test"])
            negative_examples_test = set(examples["negative_examples_test"])
            typed_pos_test = set(
                map(OWLNamedIndividual, map(IRI.create, positive_examples_test))
            )
            typed_neg_test = set(
                map(OWLNamedIndividual, map(IRI.create, negative_examples_test))
            )
            predictions = model.predict(
                individuals=list(typed_pos_test | typed_neg_test), hypotheses=hypotheses
            )
            best_concept = hypotheses[0].concept
            # concept_ind = set(
            #     [
            #         indv.get_iri().as_str()
            #         for indv in target_kb.individuals_set(best_concept)
            #     ]
            # )
            concept_length = target_kb.concept_len(hypotheses[0].concept)
            # concept_ind = concept_ind.intersection(
            #     positive_examples | negative_examples
            # )
            pos_preds = list(
                predictions[predictions.columns[0]][
                    predictions[predictions.columns[0]] == 1.0
                ].index
            )
            if kg == "mutag":
                pos = [item.split("#")[1] for item in positive_examples_test]
                neg = [item.split("#")[1] for item in negative_examples_test]
            if kg == "aifb":
                pos = [item.split("/")[-1] for item in positive_examples_test]
                neg = [item.split("/")[-1] for item in negative_examples_test]

            target_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
                "concept_individuals": pos_preds,
                "positive_examples": pos,
                "negative_examples": neg,
            }
            # print(target_dict)
        # Define the filename where you want to save the JSON
        file_path = f"results/predictions/EVO/{kg}_gnn_preds_train_test.json"
        # Check if the file exists
        if os.path.exists(file_path):
            # Remove the file
            os.remove(file_path)

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
        t1 = time.time()
        dur = t1 - t0
        print(
            f"Trained EvoLearner Train-test-split on  (fidelity) {kg} dataset on {dur : .2f}"
        )
