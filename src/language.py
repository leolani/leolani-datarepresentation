from random import getrandbits
from datetime import datetime
from typing import List
import logging
import enum

from nltk import pos_tag

from utils import casefold_text
from representation import Triple, Perspective
from rdf_builder import RdfBuilder

logger = logging.getLogger(__name__)


class Time(enum.Enum):
    """
    This will be used in the future to represent tense
    """
    PAST = -1
    PRESENT = 0
    FUTURE = 1


class Emotion(enum.Enum):
    """
    This will be used in the future to represent emotion
    """
    ANGER = 0
    DISGUST = 1
    FEAR = 2
    HAPPINESS = 3
    SADNESS = 4
    SURPRISE = 5


class UtteranceType(enum.Enum):
    STATEMENT = 0
    QUESTION = 1
    EXPERIENCE = 2


class UtteranceHypothesis(object):
    """
    Automatic Speech Recognition (ASR) Hypothesis

    Parameters
    ----------
    transcript: str
        Utterance Hypothesis Transcript
    confidence: float
        Utterance Hypothesis Confidence
    """

    def __init__(self, transcript, confidence):
        # type: (str, float) -> None

        self._transcript = transcript
        self._confidence = confidence

    @property
    def transcript(self):
        # type: () -> str
        """
        Automatic Speech Recognition Hypothesis Transcript

        Returns
        -------
        transcript: str
        """
        return self._transcript

    @transcript.setter
    def transcript(self, value):
        # type: (str) -> None
        self._transcript = value

    @property
    def confidence(self):
        # type: () -> float
        """
        Automatic Speech Recognition Hypothesis Confidence

        Returns
        -------
        confidence: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        # type: (float) -> None
        self._confidence = value

    def __repr__(self):
        return "<'{}' [{:3.2%}]>".format(self.transcript, self.confidence)


class Chat(object):
    def __init__(self, speaker, context):
        """
        Create Chat

        Parameters
        ----------
        speaker: str
            Name of speaker (a.k.a. the person Pepper has a chat with)
        context: Context
            Context Chat is part of
        """

        self._id = getrandbits(128)
        self._context = context
        self._speaker = speaker
        self._utterances = []

        self._log = self._update_logger()
        self._log.info("<< Start of Chat with {} >>".format(speaker))

    @property
    def context(self):
        """
        Returns
        -------
        context: Context
            Context
        """
        return self._context

    @property
    def speaker(self):
        # type: () -> str
        """
        Returns
        -------
        speaker: str
            Name of speaker (a.k.a. the person Pepper has a chat with)
        """
        return self._speaker

    @speaker.setter
    def speaker(self, value):
        self._speaker = value

    @property
    def id(self):
        # type: () -> int
        """
        Returns
        -------
        id: int
            Unique (random) identifier of this chat
        """
        return self._id

    @property
    def utterances(self):
        # type: () -> List[Utterance]
        """
        Returns
        -------
        utterances: list of Utterance
            List of utterances that occurred in this chat
        """
        return self._utterances

    @property
    def last_utterance(self):
        # type: () -> Utterance
        """
        Returns
        -------
        last_utterance: Utterance
            Most recent Utterance
        """
        return self._utterances[-1]

    def add_utterance(self, hypotheses, me):
        # type: (List[UtteranceHypothesis], bool) -> Utterance
        """
        Add Utterance to Conversation

        Parameters
        ----------
        hypotheses: list of UtteranceHypothesis

        Returns
        -------
        utterance: Utterance
        """
        utterance = Utterance(self, hypotheses, me, len(self._utterances))
        self._utterances.append(utterance)

        self._log = self._update_logger()
        self._log.info(utterance)

        return utterance

    def _update_logger(self):
        return logger.getChild("Chat {:19s} {:03d}".format("({})".format(self.speaker), len(self._utterances)))

    def __repr__(self):
        return "\n".join([str(utterance) for utterance in self._utterances])


class Utterance(object):
    def __init__(self, chat, hypotheses, me, turn):
        # type: (Chat, List[UtteranceHypothesis], bool, int) -> Utterance
        """
        Construct Utterance Object

        Parameters
        ----------
        chat: Chat
            Reference to Chat Utterance is part of
        hypotheses: List[UtteranceHypothesis]
            Hypotheses on uttered text (transcript, confidence)
        me: bool
            True if Robot spoke, False if Person Spoke
        turn: int
            Utterance Turn
        """

        self._log = logger.getChild(self.__class__.__name__)

        self._datetime = datetime.now()
        self._chat = chat
        self._context = self._chat.context
        self._chat_speaker = self._chat.speaker
        self._turn = turn
        self._me = me

        self._hypothesis = self._choose_hypothesis(hypotheses)
        self._tokens = self._clean(self._tokenize(self.transcript))

        self._parser = None
        self._type = None
        self._triple = None
        self._perspective = None

    @property
    def chat(self):
        # type: () -> Chat
        """
        Returns
        -------
        chat: Chat
            Utterance Chat
        """
        return self._chat

    @property
    def context(self):
        # type: () -> Context
        """
        Returns
        -------
        context: Context
            Context (a.k.a. people, objects and other detections )
        """
        return self._context

    @property
    def chat_speaker(self):
        # type: () -> str
        """
        Returns
        -------
        speaker: str
            Name of speaker (a.k.a. the person Pepper has a chat with)
        """
        return self._chat_speaker

    @property
    def type(self):
        # type: () -> UtteranceType
        """
        Returns
        -------
        type: UtteranceType
            Whether the utterance was a statement, a question or an experience
        """
        return self._type

    @property
    def transcript(self):
        # type: () -> str
        """
        Returns
        -------
        transcript: str
            Utterance Transcript
        """
        return self._hypothesis.transcript

    @property
    def confidence(self):
        # type: () -> float
        """
        Returns
        -------
        confidence: float
            Utterance Confidence
        """
        return self._hypothesis.confidence

    @property
    def me(self):
        # type: () -> bool
        """
        Returns
        -------
        me: bool
            True if Robot spoke, False if Person Spoke
        """
        return self._me

    @property
    def turn(self):
        # type: () -> int
        """
        Returns
        -------
        turn: int
            Utterance Turn
        """
        return self._turn

    @property
    def triple(self):
        # type: () -> Triple
        """
        Returns
        -------
        triple: Triple
            Structured representation of the utterance
        """
        return self._triple

    @property
    def perspective(self):
        # type: () -> Perspective
        """
        Returns
        -------
        perspective: Perspective
            NLP features related to the utterance
        """
        return self._perspective

    @property
    def datetime(self):
        return self._datetime

    @property
    def language(self):
        """
        Returns
        -------
        language: str
            Original language of the Transcript
        """
        raise NotImplementedError()

    @property
    def tokens(self):
        """
        Returns
        -------
        tokens: list of str
            Tokenized transcript
        """
        return self._tokens

    @property
    def parser(self):
        # type: () -> Optional[Parser]
        """
        Returns
        -------
        parsed_tree: ntlk Tree generated by the CFG parser
        """
        return self._parser

    def pack_triple(self, rdf, utterance_type):
        """
        Sets utterance type, the extracted triple and (in future) the perspective
        Parameters
        ----------
        rdf
        utterance_type

        Returns
        -------

        """
        self._type = utterance_type
        if type(rdf) == str:
            return rdf

        if not rdf:
            return 'error in the rdf'

        builder = RdfBuilder()

        # Build each element
        subject = builder.fill_entity(casefold_text(rdf['subject']['text'], format='triple'),
                                      rdf['subject']['type'])
        predicate = builder.fill_predicate(casefold_text(rdf['predicate']['text'], format='triple'))
        complement = builder.fill_entity(casefold_text(rdf['complement']['text'], format='triple'),
                                         rdf['complement']['type'])

        self.set_triple(Triple(subject, predicate, complement))

    def pack_perspective(self, persp):
        self.set_perspective(Perspective(persp['certainty'], persp['polarity'], persp['sentiment']))

    def set_triple(self, triple):
        # type: (Triple) -> ()
        self._triple = triple

    def set_perspective(self, perspective):
        # type: (Perspective) -> ()
        self._perspective = perspective

    def casefold(self, format='triple'):
        # type (str) -> ()
        """
        Format the labels to match triples or natural language
        Parameters
        ----------
        format

        Returns
        -------

        """
        self._triple.casefold(format)
        self._chat_speaker = casefold_text(self.chat_speaker, format)

    def _choose_hypothesis(self, hypotheses):
        return sorted(hypotheses, key=lambda hypothesis: hypothesis.confidence, reverse=True)[0]

    def _tokenize(self, transcript):
        """
        Parameters
        ----------
        transcript: str
            Uttered text (Natural Language)

        Returns
        -------
        tokens: list of str
            Tokenized transcript: list of cleaned tokens for POS tagging and syntactic parsing
                - removes contractions and openers/introductions
        """

        # possible openers/greetings/introductions are removed from the beginning of the transcript
        # it is done like this to avoid lowercasing the transcript as caps are useful and google puts them
        openers = ['Leolani', 'Sorry', 'Excuse me', 'Hey', 'Hello', 'Hi']
        introductions = ['Can you tell me', 'Do you know', 'Please tell me', 'Do you maybe know']

        for o in openers:
            if transcript.startswith(o):
                transcript = transcript.replace(o, '')
            if transcript.startswith(o.lower()):
                transcript = transcript.replace(o.lower(), '')

        for i in introductions:
            if transcript.startswith(i):
                tmp = transcript.replace(i, '')
                first_word = tmp.split()[0]
                if first_word in ['what', 'that', 'who', 'when', 'where', 'which']:
                    transcript = transcript.replace(i, '')
            if transcript.startswith(i.lower()):
                tmp = transcript.replace(i.lower(), '')
                first_word = tmp.split()[0]
                if first_word.lower() in ['what', 'that', 'who', 'when', 'where', 'which']:
                    transcript = transcript.replace(i.lower(), '')

        # separating typical contractions
        tokens_raw = transcript.replace("'", " ").split()
        dict = {'m': 'am', 're': 'are', 'll': 'will'}
        dict_not = {'won': 'will', 'don': 'do', 'doesn': 'does', 'didn': 'did', 'haven': 'have', 'wouldn': 'would',
                    'aren': 'are'}

        for key in dict:
            tokens_raw = self.replace_token(tokens_raw, key, dict[key])

        if 't' in tokens_raw:
            tokens_raw = self.replace_token(tokens_raw, 't', 'not')
            for key in dict_not:
                tokens_raw = self.replace_token(tokens_raw, key, dict_not[key])

        # in case of possessive genitive the 's' is just removed, while for the aux verb 'is' is inserted
        if 's' in tokens_raw:
            index = tokens_raw.index('s')
            try:
                tag = pos_tag([tokens_raw[index + 1]])
                if tag[0][1] in ['DT', 'JJ', 'IN'] or tag[0][1].startswith('V'):  # determiner, adjective, verb
                    tokens_raw.remove('s')
                    tokens_raw.insert(index, 'is')
                else:
                    tokens_raw.remove('s')
            except:
                tokens_raw.remove('s')

        return tokens_raw

    def replace_token(self, tokens_raw, old, new):
        '''
        :param tokens_raw: list of tokens
        :param old: token to replace
        :param new: new token
        :return: new list with the replaced token
        '''
        if old in tokens_raw:
            index = tokens_raw.index(old)
            tokens_raw.remove(old)
            tokens_raw.insert(index, new)
        return tokens_raw

    def _clean(self, tokens):
        """
        Parameters
        ----------
        tokens: list of str
            Tokenized transcript

        Returns
        -------
        cleaned_tokens: list of str
            Tokenized & Cleaned transcript
        """
        return tokens

    def __repr__(self):
        author = self.chat.context.own_name if self.me else self.chat.speaker
        return '{:>10s}: "{}"'.format(author, self.transcript)
