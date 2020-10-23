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


capsule_knows = {
    "utterance": "dimitris knows piek",
    "subject": {
        "label": "dimitris",
        "type": "person"
    },
    "predicate": {
        "type": "knows"
    },
    "object": {
        "label": "piek",
        "type": "person"
    },
    "author": "tom",
    "turn": 1,
    "position": "0-25",
    "date": date(2019, 1, 24)
}

if __name__ == "__main__":
    capsule = transform_capsule()
