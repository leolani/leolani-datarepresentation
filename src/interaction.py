import os
import re
import platform
import requests
import pycountry
import subprocess
from time import time
from datetime import datetime
from random import getrandbits
from typing import List, Iterable, Dict, Tuple, Optional

from vision import Object, Face, Observations
from language import Chat


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
