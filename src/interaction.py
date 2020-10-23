from pepper.framework.infra.util import Bounds, spherical2cartesian
from pepper import ObjectDetectionTarget

import numpy as np

from socket import socket, error as socket_error
from random import getrandbits
import json

from typing import List, Tuple, Dict


class Context(object):
    """
    Context Object

    Contains Awareness for People, Objects & Conversations
    """

    OBSERVATION_TIMEOUT = 60

    _people = None  # type: Dict[str, Tuple[Face, float]]
    _objects = None  # type: Observations

    def __init__(self, name, friends):
        # type: (str, Iterable[str]) -> None
        self._id = getrandbits(128)

        self._name = name
        self._friends = friends

        self._chats = []
        self._chat_start = None
        self._chatting = False

        self._people = {}
        self._current_people = []
        self._objects = Observations()
        self._intention = None

        self._location = Location()

    @property
    def id(self):
        # type: () -> int
        """
        ID

        Returns
        -------
        id: int
        """
        return self._id

    @property
    def own_name(self):
        # type: () -> str
        """
        Returns
        -------
        str
            The robot's own name
        """
        return self._name

    @property
    def chats(self):
        # type: () -> List[Chat]
        """
        Returns
        -------
        chats: list of Chat
            List of all Chats that were held during current session
        """
        return self._chats

    @property
    def chatting(self):
        # type: () -> bool
        """
        Returns True when a Chat is happening

        Returns
        -------
        chatting: bool
        """
        return self._chatting

    @property
    def chat(self):
        # type: () -> Optional[Chat]
        """
        The Current Chat, if any

        Returns
        -------
        chat: Optional[Chat]
        """
        return self.chats[-1] if self.chatting else None

    @property
    def datetime(self):  # When
        # type: () -> datetime
        """
        The Current Date & Time

        Returns
        -------
        datetime: datetime
            Current Date and Time
        """
        return datetime.now()

    @property
    def location(self):  # Where
        # type: () -> Location
        """
        The Current Location

        Returns
        -------
        location: Location
            Current Location
        """
        return self._location

    @property
    def people(self):  # Who
        # type: () -> List[Face]
        """
        People seen within Observation Timeout

        Returns
        -------
        people: list of Face
            List of People seen within Observation Timeout
        """
        current_time = time()

        return [person for person, t in self._people.values() if (current_time - t) < Context.OBSERVATION_TIMEOUT]

    @property
    def friends(self):
        # type: () -> List[str]
        """
        Names of all friends.

        Returns
        -------
        List[str]
            List of all friends names
        """
        return self._friends

    def current_people(self, in_chat=False, timeout=OBSERVATION_TIMEOUT):
        # type: () -> List[Face]
        """
        People seen currently in the Context

        Returns
        -------
        people: list of Face
            List of all People seen currently in the Context
        """
        if in_chat and not self.chatting:
            return []

        current_time = time()

        return [person for person, t in self._people.values()
                if current_time - t <= timeout and (not in_chat or t >= self._chat_start)]

    @property
    def all_people(self):
        # type: () -> List[Face]
        """
        People seen since beginning of Context

        Returns
        -------
        people: list of Face
            List of all People seen since beginning of Context
        """
        return [person for person, t in self._people.values()]

    @property
    def objects(self):  # What
        # type: () -> List[Object]
        """
        Objects seen within Observation Timeout

        Returns
        -------
        objects: list of Object
            List of Objects seen within Observation Timeout
        """
        return self._objects.instances

    @property
    def all_objects(self):
        # type: () -> List[Object]
        """
        Objects seen since beginning of Context

        Returns
        -------
        objects: list of Object
            List of all Objects since beginning of Context
        """
        return self._objects.instances

    def add_objects(self, objects):
        # type: (List[Object]) -> None
        """
        Add Object Observations to Context

        Parameters
        ----------
        objects: list of Object
            List of Objects
        """
        if objects:
            self._objects.add_observations(objects[0].image, objects)

    def add_people(self, people):
        # type: (Iterable[Face]) -> None
        """
        Add People Observations to Context

        Parameters
        ----------
        people: list of Face
            List of People
        """
        for person in people:
            self._people[person.name] = (person, time())

    def start_chat(self, speaker):
        # type: (str) -> None
        """
        Start Chat with Speaker

        Parameters
        ----------
        speaker: str
            Name of Speaker
        """
        self._chat_start = time()
        self._chatting = True
        self._chats.append(Chat(speaker, self))

    def stop_chat(self):
        # type: () -> None
        """Stop Chat"""
        self._chat_start = None
        self._chatting = False


class Location(object):
    """Location on Earth"""

    UNKNOWN = "Unknown"

    def __init__(self):
        # TODO use UUIDs
        self._id = getrandbits(128)
        self._label = self.UNKNOWN

        try:
            loc = requests.get("https://ipinfo.io").json()

            self._country = pycountry.countries.get(alpha_2=loc['country']).name
            self._region = loc['region']
            self._city = loc['city']
        except:
            self._country = self.UNKNOWN
            self._region = self.UNKNOWN
            self._city = self.UNKNOWN

    @property
    def id(self):
        # type: () -> int
        """
        ID for this Location object

        Returns
        -------
        id: int
        """
        return self._id

    @property
    def country(self):
        # type: () -> str
        """
        Country String

        Returns
        -------
        country: str
        """
        return self._country

    @property
    def region(self):
        # type: () -> str
        """
        Region String

        Returns
        -------
        region: str
        """
        return self._region

    @property
    def city(self):
        # type: () -> str
        """
        City String

        Returns
        -------
        city: str
        """
        return self._city

    @property
    def label(self):
        # type: () -> str
        """
        Learned Location Label

        Returns
        -------
        label: str
        """
        return self._label

    @label.setter
    def label(self, value):
        # type: (str) -> None
        """
        Learned Location Label

        Parameters
        ----------
        value: str
        """
        self._label = value

    @staticmethod
    def _get_lat_lon():
        # type: () -> Optional[Tuple[float, float]]
        """
        Get Latitude & Longitude from GPS

        Returns
        -------
        latlon: Optional[Tuple[float, float]]
            GPS Latitude & Longitude
        """
        try:
            if platform.system() == "Darwin":
                # Use WhereAmI tool by Rob Mathers -> https://github.com/robmathers/WhereAmI
                whereami = os.path.join(os.path.dirname(__file__), 'util', 'whereami')
                regex = "Latitude: (.+?)\nLongitude: (.+?)\n"
                return tuple(float(coord) for coord in re.findall(regex, subprocess.check_output(whereami))[0])
            else:
                raise Exception()
        except:  # TODO: Add Support for (at least) Windows
            print("Couldn't get GPS Coordinates")
            return None

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.city, self.region, self.country)


class Observations:
    """
    Object Observations

    Object to track of which objects have been seen and where. Groups Object Observations based on location,
    to guess which observations are in fact of the same object. Each Object Class is handled in ObjectObservations.
    """

    def __init__(self):
        # type: () -> None
        self._object_observations = {}

    @property
    def instances(self):
        # type: () -> List[Object]
        """
        Get individual object instances, based on observations

        Returns
        -------
        instances: List[Object]
        """
        instances = []

        for object_observations in self._object_observations.values():
            instances.extend(object_observations.instances)

        return instances

    def add_observations(self, image, objects):
        # type: (AbstractImage, List[Object]) -> None
        """
        Add Object Observations and figure out with which Object Instance they correspond

        Parameters
        ----------
        image: AbstractImage
        objects: List[Object]
        """
        for obj in objects:
            if obj.name not in self._object_observations:
                self._object_observations[obj.name] = ObjectObservations(obj.name)
            self._object_observations[obj.name].add_observation(obj)

        for object_observations in self._object_observations.values():
            object_observations.update_view(image)


class Object(object):
    """
    'Object' object

    Parameters
    ----------
    name: str
        Name of Object
    confidence: float
        Object Name & Bounds Confidence
    bounds: Bounds
        Bounds in Image Space
    image: AbstractImage
        Image from which Object was Recognised
    """

    def __init__(self, name, confidence, bounds, image):
        # type: (str, float, Bounds, AbstractImage) -> None
        self._id = getrandbits(128)

        self._name = name
        self._confidence = confidence
        self._image_bounds = bounds
        self._image = image

        # Calculate Position in 2D Angular Space (Phi, Theta)
        self._bounds = self._calculate_bounds()
        self._direction = self.bounds.center

        # Calculate Position in 3D Space (Relative to Robot)
        self._depth = self._calculate_object_depth()
        self._position = spherical2cartesian(self._direction[0], self._direction[1], self._depth)
        self._bounds3D = self._calculate_bounds_3D()

    @classmethod
    def from_json(cls, data, image):
        # type: (Dict, AbstractImage) -> Object
        return cls(data["name"], data["confidence"], Bounds.from_json(data["bounds"]), image)

    @property
    def id(self):
        # type: () -> int
        """
        Object ID

        Returns
        -------
        id: int
        """
        return self._id

    @property
    def name(self):
        # type: () -> str
        """
        Object Name

        Returns
        -------
        name: str
            Name of Person
        """
        return self._name

    @property
    def confidence(self):
        # type: () -> float
        """
        Object Confidence

        Returns
        -------
        confidence: float
            Object Name & Bounds Confidence
        """
        return self._confidence

    @property
    def time(self):
        # type: () -> float
        """
        Time of Observation

        Returns
        -------
        time: float
        """
        try:
            return self.image.time
        except:
            return 0.0

    @property
    def image_bounds(self):
        # type: () -> Bounds
        """
        Object Bounds in Image Space {x: [0, 1], y: [0, 1]}

        Returns
        -------
        bounds: Bounds
            Object Bounding Box in Image Space
        """
        return self._image_bounds

    @property
    def bounds(self):
        # type: () -> Bounds
        """
        Object Bounds in View Space {x: [-pi, +pi], y: [0, pi]}

        Returns
        -------
        bounds: Bounds
            Object Bounding Box in View Space
        """
        return self._bounds

    @property
    def image(self):
        # type: () -> AbstractImage
        """
        Image associated with the observation of this Object

        Returns
        -------
        image: AbstractImage
        """
        return self._image

    @property
    def direction(self):
        # type: () -> Tuple[float, float]
        """
         Direction of Object in View Space (equivalent to self.bounds.center)

        Returns
        -------
        direction: float, float
            Direction of Object in View Space
        """
        return self._direction

    @property
    def depth(self):
        # type: () -> float
        """
        Distance from Camera to Object

        Returns
        -------
        depth: float
            Distance from Camera to Object
        """
        return self._depth

    @property
    def position(self):
        # type: () -> Tuple[float, float, float]
        """
        Position of Object in Cartesian Coordinates (x,y,z), Relative to Camera

        Returns
        -------
        position: Tuple[float, float, float]
            Position of Object in Cartesian Coordinates (x,y,z)
        """
        return self._position

    @property
    def bounds3D(self):
        # type: () -> List[Tuple[float, float, float]]
        """
        3D bounds (for visualisation) [x,y,z]*4

        Returns
        -------
        bounds3D: List[Tuple[float, float, float]]
            3D bounds (for visualisation) [x,y,z]*4
        """
        return self._bounds3D

    def distance_to(self, obj):
        # type: (Object) -> float
        """
        Distance from this Object to obj

        Parameters
        ----------
        obj: Object

        Returns
        -------
        distance: float
        """
        return np.sqrt(
            (self.position[0] - obj.position[0]) ** 2 +
            (self.position[1] - obj.position[1]) ** 2 +
            (self.position[2] - obj.position[2]) ** 2
        )

    def dict(self):
        # type: () -> Dict
        """
        Object to Dictionary

        Returns
        -------
        dict: Dict
            Dictionary representation of Object
        """

        return {
            "name": self.name,
            "confidence": self.confidence,
            "bounds": self.image_bounds.dict(),
            "image": self.image.hash
        }

    def json(self):
        # type: () -> str
        """
        Object to JSON

        Returns
        -------
        json: JSON representation of Object
        """
        return json.dumps(self.dict())

    def _calculate_object_depth(self):
        # type: () -> float
        """
        Calculate Distance of Object to Camera
        Take the median of all valid depth pixels...

        # TODO: Improve Depth Calculation

        Returns
        -------
        depth: float
        """
        try:
            depth_map = self.image.get_depth(self._image_bounds)
            depth_map_valid = depth_map != 0

            if np.sum(depth_map_valid):
                return np.median(depth_map[depth_map_valid])
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_bounds(self):
        # type: () -> Bounds
        """
        Calculate View Space Bounds from Image Space Bounds

        Returns
        -------
        bounds: Bounds
            Bounds in View Space
        """
        try:
            x0, y0 = self._image.get_direction((self._image_bounds.x0, self._image_bounds.y0))
            x1, y1 = self._image.get_direction((self._image_bounds.x1, self._image_bounds.y1))
            return Bounds(x0, y0, x1, y1)
        except:
            return Bounds(0, 0, 0, 0)

    def _calculate_bounds_3D(self):
        # type: () -> List[List[float]]
        """
        Calculate 3D Bounds (for visualisation)

        Returns
        -------
        bounds_3D: List[List[float]]
        """
        return [
            spherical2cartesian(self._bounds.x0, self._bounds.y0, self._depth),
            spherical2cartesian(self._bounds.x0, self._bounds.y1, self._depth),
            spherical2cartesian(self._bounds.x1, self._bounds.y1, self._depth),
            spherical2cartesian(self._bounds.x1, self._bounds.y0, self._depth),
        ]

    def __repr__(self):
        return "{}({}, {:3.0%})".format(self.__class__.__name__, self.name, self.confidence)


class Face(Object):
    """
    Face Object

    Parameters
    ----------
    name: str
        Name of Person
    confidence: float
        Name Confidence
    representation: np.ndarray
        Face Feature Vector
    bounds: Bounds
        Face Bounding Box
    image: AbstractImage
        Image Face was Found in
    """

    UNKNOWN = config.HUMAN_UNKNOWN

    def __init__(self, name, confidence, representation, bounds, image):
        # type: (str, float, np.ndarray, Bounds, AbstractImage) -> None
        super(Face, self).__init__(Face.UNKNOWN if name == FaceClassifier.NEW else name,
                                   confidence, bounds, image)

        self._representation = representation

    @property
    def representation(self):
        """
        Face Representation

        Returns
        -------
        representation: np.ndarray
            Face Feature Vector
        """
        return self._representation
