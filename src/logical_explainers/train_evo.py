import json

from ontolearn.concept_learner import EvoLearner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import IRI, OWLDeclarationAxiom, OWLNamedIndividual


def train_evo(file_path=None):
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

        model = EvoLearner(knowledge_base=target_kb, max_runtime=600)
        model.fit(lp, verbose=False)

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
        file_path = f"results/actuals/EVO/{kg}.json"

        # Save the dictionary to a JSON file with indentation
        with open(file_path, "w") as json_file:
            json.dump(target_dict, json_file, indent=4)
