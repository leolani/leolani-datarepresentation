from constants import CAPITALIZED_TYPES


def hash_claim_id(triple):
    return '_'.join(triple)


def is_proper_noun(types):
    return any(i in types for i in CAPITALIZED_TYPES)


def casefold_text(text, format='triple'):
    if format == 'triple':
        return text.strip().lower().replace(" ", "-") if isinstance(text, basestring) else text
    elif format == 'natural':
        return text.strip().lower().replace("-", " ") if isinstance(text, basestring) else text
    else:
        return text
