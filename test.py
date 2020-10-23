from src import *


def fake_context():
    faces = {Face('Selene', 0.90, None, None, None), Face('Piek', 0.90, None, None, None)}

    context = Context("Leolani", [])
    context.location._label = 'Office'

    objects = [Object('person', 0.79, None, None), Object('laptop', 0.88, None, None),
               Object('chair', 0.88, None, None), Object('laptop', 0.51, None, None),
               Object('teddy bear', 0.88, None, None)]

    context.add_objects(objects)
    context.add_people(faces)

    return context


def transform_capsule(capsule):
    context = fake_context()

    chat = Chat(capsule['author'], context)
    hyp = UtteranceHypothesis(capsule['utterance'], 0.99)

    utt = Utterance(chat, [hyp], False, capsule['turn'])
    utt._type = UtteranceType.STATEMENT

    builder = RdfBuilder()

    triple = builder.fill_triple(capsule['subject'], capsule['predicate'], capsule['object'])
    utt.set_triple(triple)

    utt.pack_perspective(capsule['perspective'])

    return utt


if __name__ == "__main__":
    capsule_knows = {
        "utterance": "I think Lenka is from Serbia",
        "subject": {"label": "lenka", "type": "person"},
        "predicate": {"type": "be-from"},
        "object": {"label": "serbia", "type": "location"},
        "perspective": {"certainty": 0.5, "polarity": 1, "sentiment": 0},
        "author": "piek",
        "chat": 1,
        "turn": 1,
        "position": "0-25",
        "date": date(2018, 3, 19)
    }

    utterance = transform_capsule(capsule_knows)
