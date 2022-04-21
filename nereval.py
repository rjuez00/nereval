#https://github.com/jantrienes/nereval
#modified version off

# pylint: disable=C0103
from __future__ import division
import argparse, collections, json, pandas as pd

Entity = collections.namedtuple('Entity', ['text', 'type', 'start'])

def has_overlap(x, y):
    """
    Determines whether the text of two entities overlap. This function is symmetric.

    Returns
    -------
    bool
        True iff text overlaps.
    """
    end_x = x.start + len(x.text)
    end_y = y.start + len(y.text)
    return x.start < end_y and y.start < end_x

def percentage_overlap_on_x(x,y):
    end_x = x.start + len(x.text)
    end_y = y.start + len(y.text)
    a = x
    x = set(range(x.start, end_x))
    if len(x) == 0:
        print(a)


    y = set(range(y.start, end_y))
    return len(x.intersection(y))/len(x)


def correct_text(x, y):
    """
    Assert entity boundaries are correct regardless of entity type.
    """
    return x.text == y.text and x.start == y.start

def correct_type(x, y):
    """
    Assert entity types match and that there is an overlap in the text of the two entities.
    """
    return x.type == y.type and has_overlap(x, y)

def count_correct(true, pred, gold_coverage = 0.8, debug = False):
    """
    Computes the count of correctly predicted entities on two axes: type and text.

    Parameters
    ----------
    true: list of Entity
        The list of ground truth entities.
    pred: list of Entity
        The list of predicted entities.

    Returns
    -------
    count_text: int
        The number of entities predicted where the text matches exactly.
    count_lower_coverage: int
        The number of entities where the type is correctly predicted and the text overlaps.
    """
    count_text, count_lower_coverage = 0, 0

    for x in true:
        for y in pred:
            text_match = correct_text(x, y)
            type_match = percentage_overlap_on_x(x, y) >= gold_coverage

            if text_match:
                if debug == True:
                    print("COMPLETE MATCH:\n\t", x.text, " - ", y.text, "\n\t", x.start,y.start)
                count_text += 1

            if type_match:
                count_lower_coverage += 1
                if not text_match and debug == True:
                    print("PARTIAL MATCH:\n\t", x.text, " - ", y.text, "\n\t", f"{x.start} - {x.start + len(x.text)}", f"{y.start} - {y.start + len(y.text)}")




            if type_match or text_match:
                # Stop as soon as an entity has been recognized by the system
                break
    
    return count_text, count_lower_coverage

def precision(correct, actual):
    if actual == 0:
        return 0

    return correct / actual

def recall(correct, possible):
    if possible == 0:
        return 0

    return correct / possible

def f1(p, r):
    if p + r == 0:
        return 0

    return 2 * (p * r) / (p + r)

def get_counts_correct(y_true, y_pred, gold_coverage = 0.8, debug = False):
    
    x, y = y_true, y_pred


    count_text, count_lower_coverage = count_correct(x, y, gold_coverage=gold_coverage, debug = debug)
    correctComplete = count_text
    correctPartial  = count_lower_coverage
    possible = len(x) 
    actual = len(y) 

    return correctComplete, correctPartial, possible, actual


def evaluate_document(y_true, y_pred, gold_coverage = 0.8, debug = False):
    """
    Evaluate classification results for a whole dataset. Each row corresponds to one text in the
    dataset.

    Parameters
    ----------
    y_true: list of list
        a list of ground-truth entities.
    y_pred: list of list
        a list of predicted entities.

    Returns
    -------
    float:
        Micro-averaged F1 score of precision and recall, precision and recall.

    Example
    -------
    >>> from nereval import Entity, evaluate
    >>> y_true = [
    ...     [Entity('a', 'b', 0), Entity('b', 'b', 2)]
    ... ]
    >>> y_pred = [
    ...     [Entity('b', 'b', 2)]
    ... ]
    >>> evaluate(y_true, y_pred)
    0.6666666666666666
    """
    if gold_coverage == 0:
        print("COVERAGE CANT BE 0")
        return None

    correctComplete, correctPartial, possible, actual = get_counts_correct(y_true, y_pred, gold_coverage=gold_coverage, debug = debug)


    calculatedPrecisionComplete = precision(correctComplete, actual)
    calculatedRecallComplete = recall(correctComplete, possible)
    scoresComplete = (f1(calculatedPrecisionComplete, calculatedRecallComplete), calculatedPrecisionComplete, calculatedRecallComplete)
    
    calculatedPrecisionPartial = precision(correctPartial, actual)
    calculatedRecallPartial = recall(correctPartial, possible)
    scoresPartial = (f1(calculatedPrecisionPartial, calculatedRecallPartial), calculatedPrecisionPartial, calculatedRecallPartial)
    
    return scoresComplete, scoresPartial



def evaluate_dataset(zip_of_y_true_pred, gold_coverage = 0.8, debug = False):
    correctComplete, correctPartial, possible, actual = 0, 0, 0, 0

    for y_true, y_pred in zip_of_y_true_pred:
        partialcorrectComplete, partialcorrectPartial, partialpossible, partialactual = get_counts_correct(y_true, y_pred, gold_coverage=gold_coverage, debug = debug)
        correctComplete += partialcorrectComplete
        correctPartial += partialcorrectPartial
        possible += partialpossible
        actual += partialactual
        


    calculatedPrecisionComplete = precision(correctComplete, actual)
    calculatedRecallComplete = recall(correctComplete, possible)
    scoresComplete = (f1(calculatedPrecisionComplete, calculatedRecallComplete), calculatedPrecisionComplete, calculatedRecallComplete)
    
    calculatedPrecisionPartial = precision(correctPartial, actual)
    calculatedRecallPartial = recall(correctPartial, possible)
    scoresPartial = (f1(calculatedPrecisionPartial, calculatedRecallPartial), calculatedPrecisionPartial, calculatedRecallPartial)

    return scoresComplete, scoresPartial


def pretty_print_scores(scores):
    pd.set_option("display.precision", 4)
    columns = pd.MultiIndex.from_tuples(
        [(libraryName, label, coverage) for libraryName in scores.keys() for label in scores[libraryName]   for coverage in ["Complete", "Partial"] ], 
        names=["Method", "Coverage", "Entity Type"]
        )

    df = pd.DataFrame(index = ["F1", "Precision", "Recall"], columns=columns)

    for libraryName in scores.keys():
            for label in scores[libraryName]:
                df.loc[:,(libraryName, label, "Complete")] = scores[libraryName][label][0]
                df.loc[:,(libraryName, label, "Partial")] = scores[libraryName][label][1]



    return df



def _parse_json(file_name):
    data = None

    with open(file_name) as json_file:
        data = json.load(json_file)

        dict_to_entity = lambda e: Entity(e['text'], e['type'], e['start'])
        for instance in data:
            instance['true'] = [dict_to_entity(e) for e in instance['true']]
            instance['predicted'] = [dict_to_entity(e) for e in instance['predicted']]

    return data


