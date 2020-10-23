import logging

from iribaker import to_iri
from rdflib import URIRef, Literal, Namespace

from representation import Predicate, Entity, Triple, Provenance

logger = logging.getLogger(__name__)


class RdfBuilder(object):

    def __init__(self):
        # type: () -> None

        self.namespaces = {}

        self._log = logger.getChild(self.__class__.__name__)
        self._log.debug("Booted")

        self._define_namespaces()

    ########## setting up connection ##########
    def _define_namespaces(self):
        """
        Define namespaces for different layers (ontology/vocab and resource). Assign them to self
        :return:
        """
        # Namespaces for the instance layer
        instance_vocab = 'http://cltl.nl/leolani/n2mu/'
        self.namespaces['N2MU'] = Namespace(instance_vocab)
        instance_resource = 'http://cltl.nl/leolani/world/'
        self.namespaces['LW'] = Namespace(instance_resource)

        # Namespaces for the mention layer
        mention_vocab = 'http://groundedannotationframework.org/gaf#'
        self.namespaces['GAF'] = Namespace(mention_vocab)
        mention_resource = 'http://cltl.nl/leolani/talk/'
        self.namespaces['LTa'] = Namespace(mention_resource)

        # Namespaces for the attribution layer
        attribution_vocab = 'http://groundedannotationframework.org/grasp#'
        self.namespaces['GRASP'] = Namespace(attribution_vocab)
        factuality_vocab = 'http://groundedannotationframework.org/grasp/factuality#'
        self.namespaces['GRASPf'] = Namespace(factuality_vocab)
        sentiment_vocab = 'http://groundedannotationframework.org/grasp/sentiment#'
        self.namespaces['GRASPs'] = Namespace(sentiment_vocab)
        emotion_vocab = 'http://groundedannotationframework.org/grasp/emotion#'
        self.namespaces['GRASPe'] = Namespace(emotion_vocab)
        attribution_resource_friends = 'http://cltl.nl/leolani/friends/'
        self.namespaces['LF'] = Namespace(attribution_resource_friends)
        attribution_resource_inputs = 'http://cltl.nl/leolani/inputs/'
        self.namespaces['LI'] = Namespace(attribution_resource_inputs)

        # Namespaces for the temporal layer-ish
        context_vocab = 'http://cltl.nl/episodicawareness/'
        self.namespaces['EPS'] = Namespace(context_vocab)
        self.namespaces['LC'] = Namespace('http://cltl.nl/leolani/context/')

        # The namespaces of external ontologies
        skos = 'http://www.w3.org/2004/02/skos/core#'
        self.namespaces['SKOS'] = Namespace(skos)

        prov = 'http://www.w3.org/ns/prov#'
        self.namespaces['PROV'] = Namespace(prov)

        sem = 'http://semanticweb.cs.vu.nl/2009/11/sem/'
        self.namespaces['SEM'] = Namespace(sem)

        time = 'http://www.w3.org/TR/owl-time/#'
        self.namespaces['TIME'] = Namespace(time)

        xml = 'https://www.w3.org/TR/xmlschema-2/#'
        self.namespaces['XML'] = Namespace(xml)

        wd = 'http://www.wikidata.org/entity/'
        self.namespaces['WD'] = Namespace(wd)

        wdt = 'http://www.wikidata.org/prop/direct/'
        self.namespaces['WDT'] = Namespace(wdt)

        wikibase = 'http://wikiba.se/ontology#'
        self.namespaces['wikibase'] = Namespace(wikibase)

    ########## basic constructors ##########

    def create_resource_uri(self, namespace, resource_name):
        # type: (str, str) -> str
        """
        Create an URI for the given resource (entity, predicate, named graph, etc) in the given namespace
        Parameters
        ----------
        namespace: str
            Namespace where entity belongs to
        resource_name: str
            Label of resource

        Returns
        -------
        uri: str
            Representing the URI of the resource

        """
        if namespace in self.namespaces.keys():
            uri = URIRef(to_iri(self.namespaces[namespace] + resource_name))
        else:
            uri = URIRef(to_iri('{}:{}'.format(namespace, resource_name)))

        return uri

    def fill_literal(self, value, datatype=None):
        # type: (str, str) -> Literal
        """
        Create an RDF literal given its value and datatype
        Parameters
        ----------
        value: str
            Value of the literal resource
        datatype: str
            Datatype of the literal

        Returns
        -------
            Literal with value and datatype given
        """

        return Literal(value, datatype=datatype) if datatype is not None else Literal(value)

    def fill_entity(self, label, types, namespace='LW', uri=None):
        # type: (str, list, str, str) -> Entity
        """
        Create an RDF entity given its label, types and its namespace
        Parameters
        ----------
        label: str
            Label of entity
        types: List[str]
            List of types for this entity
        uri: str
            URI of the entity, is available (i.e. when extracting concepts from wikidata)
        namespace: str
            Namespace where entity belongs to

        Returns
        -------
            Entity object with given label
        """
        if types in [None, ''] and label != '':
            self._log.warning('Unknown type: {}'.format(label))
            return self.fill_entity_from_label(label, namespace)
        else:
            entity_id = self.create_resource_uri(namespace, label) if not uri else URIRef(to_iri(uri))
            return Entity(entity_id, Literal(label), types)

    def fill_predicate(self, label, namespace='N2MU', uri=None):
        # type: (str, str, str) -> Predicate
        """
        Create an RDF predicate given its label and its namespace
        Parameters
        ----------
        label: str
            Label of predicate
        uri: str
            URI of the predicate, is available (i.e. when extracting concepts from wikidata)
        namespace:
            Namespace where predicate belongs to

        Returns
        -------

            Predicate object with given label
        """
        predicate_id = self.create_resource_uri(namespace, label) if not uri else URIRef(to_iri(uri))

        return Predicate(predicate_id, Literal(label))

    def fill_entity_from_label(self, label, namespace='LW', uri=None):
        # type: (str, str, str) -> Entity
        """
        Create an RDF entity given its label and its namespace
        Parameters
        ----------
        label: str
            Label of entity
        uri: str
            URI of the entity, is available (i.e. when extracting concepts from wikidata)
        namespace: str
            Namespace where entity belongs to

        Returns
        -------
            Entity object with given label and no type information
        """
        entity_id = self.create_resource_uri(namespace, label) if not uri else URIRef(to_iri(uri))

        return Entity(entity_id, Literal(label), [''])

    def empty_entity(self):
        # type: () -> Entity
        """
        Create an empty RDF entity
        Parameters
        ----------

        Returns
        -------
            Entity object with no label and no type information
        """
        return Entity('', Literal(''), [''])

    def fill_provenance(self, author, date):
        # type: (str, date) -> Provenance
        """
        Structure provenance to pair authors and dates when mentions are created
        Parameters
        ----------
        author: str
            Actor that generated the knowledge
        date: date
            Date when knowledge was generated

        Returns
        -------
            Provenance object containing author and date
        """

        return Provenance(author, date)

    def fill_triple(self, subject_dict, predicate_dict, object_dict, namespace='LW'):
        # type: (dict, dict, dict, str) -> Triple
        """
        Create an RDF entity given its label and its namespace
        Parameters
        ----------
        subject_dict: dict
            Information about label and type of subject
        predicate_dict: dict
            Information about type of predicate
        object_dict: dict
            Information about label and type of object
        namespace: str
            Information about which namespace the entities belongs to

        Returns
        -------
            Entity object with given label
        """
        subject = self.fill_entity(subject_dict['label'], [subject_dict['type']], namespace=namespace)
        predicate = self.fill_predicate(predicate_dict['type'])
        object = self.fill_entity(object_dict['label'], [object_dict['type']], namespace=namespace)

        return Triple(subject, predicate, object)

    def fill_triple_from_label(self, subject_label, predicate, object_label, namespace='LW'):
        # type: (str, str, str, str) -> Triple
        """
        Create an RDF entity given its label and its namespace
        Parameters
        ----------
        subject_label: str
            Information about label of subject
        predicate: str
            Information about predicate
        object_label: str
            Information about label of object
        namespace: str
            Information about which namespace the entities belongs to

        Returns
        -------
            Entity object with given label
        """
        subject = self.fill_entity_from_label(subject_label, namespace=namespace)
        predicate = self.fill_predicate(predicate)
        object = self.fill_entity_from_label(object_label, namespace=namespace)

        return Triple(subject, predicate, object)
