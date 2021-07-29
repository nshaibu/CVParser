############################################  NOTE  ########################################################
#
#           Creates NER training data in Spacy format from JSON downloaded from Dataturks.
#
#           Outputs the Spacy training data which can be used for Spacy training.
#
############################################################################################################
import json
import random
import logging
import string
import re

# from sklearn.metrics import classification_report
# from sklearn.metrics import precision_recall_fscore_support
# from spacy.gold import GoldParse
# from spacy.scorer import Scorer
# from sklearn.metrics import accuracy_score


def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end

            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
                print('end:', valid_end)
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data


def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines = []

        with open(dataturks_JSON_FilePath, 'r') as f:
            lines = f.readlines()

        for line in lines:

            data = json.loads(line)

            text = data['content']

            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1, label))


            # print(entities)
            training_data.append((text, {"entities": entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None


if __name__ == '__main__':
    INPUT_FILES = ["traindata.json", "traindata1.json"]

    TRAIN_DATA = []

    for file in INPUT_FILES:
        data = trim_entity_spans(convert_dataturks_to_spacy(file))
        TRAIN_DATA.extend(data)

    with open("output_train_data.json", "w") as fd:
        fd.write(json.dumps(TRAIN_DATA))
