import os
import json
import numpy as np
from PIL import Image
from random import getrandbits
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict
from time import time, strftime, localtime

from utils import spherical2cartesian, OBJECT_INFO


class Bounds(object):
    """
    Rectangle Bounds Object

    Parameters
    ----------
    x0: float
    y0: float
    x1: float
    y1: float
    """

    def __init__(self, x0, y0, x1, y1):
        # type: (float, float, float, float) -> None

        if x0 > x1 or y0 > y1:
            raise RuntimeWarning("Rectangle Error: Point (x1,y1) should be bigger than point (x0, y0)")

        self._x0 = x0
        self._y0 = y0
        self._x1 = x1
        self._y1 = y1

    @classmethod
    def from_json(cls, data):
        # type: (dict) -> Bounds
        """
        Create Bounds Object from Dictionary

        Parameters
        ----------
        data: dict
            Dictionary containing x0, y0, x1, y1 keys

        Returns
        -------
        bounds: Bounds
        """
        return cls(data["x0"], data["y0"], data["x1"], data["y1"])

    @property
    def x0(self):
        # type: () -> float
        """
        X0

        Returns
        -------
        x0: float
        """
        return self._x0

    @property
    def y0(self):
        # type: () -> float
        """
        Y0

        Returns
        -------
        y0: float
        """
        return self._y0

    @property
    def x1(self):
        # type: () -> float
        """
        X1

        Returns
        -------
        x1: float
        """
        return self._x1

    @property
    def y1(self):
        # type: () -> float
        """
        Y1

        Returns
        -------
        y1: float
        """
        return self._y1

    @property
    def width(self):
        # type: () -> float
        """
        Bounds Width

        Returns
        -------
        width: float
        """
        return self.x1 - self.x0

    @property
    def height(self):
        # type: () -> float
        """
        Bounds Height

        Returns
        -------
        height: float
        """
        return self.y1 - self.y0

    @property
    def center(self):
        # type: () -> (float, float)
        """
        Bounds Center

        Returns
        -------
        center: tuple
        """
        return (self.x0 + self.width / 2, self.y0 + self.height / 2)

    @property
    def area(self):
        # type: () -> float
        """
        Bounds Area

        Returns
        -------
        area: float
        """
        return self.width * self.height

    def intersection(self, bounds):
        # type: (Bounds) -> Optional[Bounds]
        """
        Bounds Intersection with another Bounds

        Parameters
        ----------
        bounds: Bounds

        Returns
        -------
        intersection: Bounds or None
        """

        x0 = max(self.x0, bounds.x0)
        y0 = max(self.y0, bounds.y0)
        x1 = min(self.x1, bounds.x1)
        y1 = min(self.y1, bounds.y1)

        return None if x0 >= x1 or y0 >= y1 else Bounds(x0, y0, x1, y1)

    def overlap(self, other):
        # type: (Bounds) -> float
        """
        Bounds Overlap Ratio

        Parameters
        ----------
        other: Bounds

        Returns
        -------
        overlap: float
        """

        intersection = self.intersection(other)

        if intersection:
            return min(intersection.area / self.area, self.area / intersection.area)
        else:
            return 0.0

    def is_subset_of(self, other):
        # type: (Bounds) -> bool
        """
        Whether 'other' Bounds is subset of 'this' Bounds

        Parameters
        ----------
        other: Bounds

        Returns
        -------
        is_subset_of: bool
            Whether 'other' Bounds is subset of 'this' Bounds
        """
        return self.x0 >= other.x0 and self.y0 >= other.y0 and self.x1 <= other.x1 and self.y1 <= other.y1

    def is_superset_of(self, other):
        # type: (Bounds) -> float
        """
        Whether 'other' Bounds is superset of 'this' Bounds

        Parameters
        ----------
        other: Bounds

        Returns
        -------
        is_superset_of: bool
            Whether 'other' Bounds is superset of 'this' Bounds
        """
        return self.x0 <= other.x0 and self.y0 <= other.y0 and self.x1 >= other.x1 and self.y1 >= other.y1

    def contains(self, point):
        # type: ((float, float)) -> bool
        """
        Whether Point lies in Bounds

        Parameters
        ----------
        point: Tuple[float, float]

        Returns
        -------
        is_in: bool
            Whether Point lies in Bounds
        """
        x, y = point
        return self.x0 < x < self.x1 and self.y0 < y < self.y1

    def equals(self, other):
        # type: (Bounds) -> bool
        """
        Whether 'other' bounds equals 'this' bounds

        Parameters
        ----------
        other: Bounds

        Returns
        -------
        equals: bool
            Whether 'other' bounds equals 'this' bounds
        """
        return self.x0 == other.x0 and self.y0 == other.y0 and self.x1 == other.x1 and self.y1 == other.y1

    def scaled(self, x_scale, y_scale):
        # type: (float, float) -> Bounds
        """
        Return Scaled Bounds Object

        Parameters
        ----------
        x_scale: float
        y_scale: float

        Returns
        -------
        bounds: Bounds
            Scaled Bounds object
        """
        return Bounds(self.x0 * x_scale, self.y0 * y_scale, self.x1 * x_scale, self.y1 * y_scale)

    def to_list(self):
        # type: () -> List[float]
        """
        Export Bounds as List

        Returns
        -------
        bounds: List[float]
        """
        return [self.x0, self.y0, self.x1, self.y1]

    def dict(self):
        # type: () -> Dict[str, float]
        """
        Export Bounds as Dict

        Returns
        -------
        dict: Dict[str, float]
        """
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1
        }

    @property
    def json(self):
        # type: () -> str
        """
        Export Bounds as JSON

        Returns
        -------
        json: str
        """
        return json.dumps(self.dict())

    def __repr__(self):
        return "Bounds[({:3f}, {:3f}), ({:3f}, {:3f})]".format(self.x0, self.y0, self.x1, self.y1)


class AbstractImage(object):
    """
    Abstract Image Container

    Parameters
    ----------
    image: np.ndarray
        RGB Image (height, width, 3) as Numpy Array
    bounds: Bounds
        Image Bounds (View Space) in Spherical Coordinates (Phi, Theta)
    depth: np.ndarray
        Depth Image (height, width) as Numpy Array
    """

    def __init__(self, image, bounds, depth=None, image_time=None):
        # type: (np.ndarray, Bounds, Optional[np.ndarray]) -> None

        self._image = image
        self._bounds = bounds
        self._depth = np.ones((100, 100), np.float32) if depth is None else depth

        self._time = image_time if image_time else time()

    @property
    def hash(self):
        return "{}_{}".format(strftime("%Y%m%d_%H%M%S", localtime(self.time)), str(self.time % 1)[2:4])

    @property
    def image(self):
        # type: () -> np.ndarray
        """
        RGB Image (height, width, 3) as Numpy Array

        Returns
        -------
        image: np.ndarray
            RGB Image (height, width, 3) as Numpy Array
        """
        return self._image

    @property
    def depth(self):
        # type: () -> Optional[np.ndarray]
        """
        Depth Image (height, width) as Numpy Array

        Returns
        -------
        depth: np.ndarray
            Depth Image (height, width) as Numpy Array
        """
        return self._depth

    @property
    def bounds(self):
        # type: () -> Bounds
        """
        Image Bounds (View Space) in Spherical Coordinates (Phi, Theta)

        Returns
        -------
        bounds: Bounds
            Image Bounds (View Space) in Spherical Coordinates (Phi, Theta)
        """
        return self._bounds

    def get_image(self, bounds):
        # type: (Bounds) -> np.ndarray
        """
        Get pixels from Image at Bounds in Image Space

        Parameters
        ----------
        bounds: Bounds
            Image Bounds (Image) in Image Space (y, x)

        Returns
        -------
        pixels: np.ndarray
            Requested pixels within Bounds
        """

        x0 = int(bounds.x0 * self._image.shape[1])
        x1 = int(bounds.x1 * self._image.shape[1])
        y0 = int(bounds.y0 * self._image.shape[0])
        y1 = int(bounds.y1 * self._image.shape[0])

        return self._image[y0:y1, x0:x1]

    def get_depth(self, bounds):
        # type: (Bounds) -> Optional[np.ndarray]
        """
        Get depth from Image at Bounds in Image Space

        Parameters
        ----------
        bounds: Bounds
            Image Bounds (Image) in Image Space (y, x)

        Returns
        -------
        depth: np.ndarray
            Requested depth within Bounds
        """

        if self._depth is None:
            return None

        x0 = int(bounds.x0 * self._depth.shape[1])
        x1 = int(bounds.x1 * self._depth.shape[1])
        y0 = int(bounds.y0 * self._depth.shape[0])
        y1 = int(bounds.y1 * self._depth.shape[0])

        return self._depth[y0:y1, x0:x1]

    def get_direction(self, coordinates):
        # type: (Tuple[float, float]) -> Tuple[float, float]
        """
        Convert 2D Image Coordinates [x, y] to 2D position in Spherical Coordinates [phi, theta]

        Parameters
        ----------
        coordinates: Tuple[float, float]

        Returns
        -------
        direction: Tuple[float, float]
        """
        return (self.bounds.x0 + coordinates[0] * self.bounds.width,
                self.bounds.y0 + coordinates[1] * self.bounds.height)

    @property
    def time(self):
        # type: () -> float
        """
        Get time image was captured and received by the application.

        Returns
        -------
        time: float
        """
        return self._time

    def frustum(self, depth_min, depth_max):
        # type: (float, float) -> List[float]
        """
        Calculate `Frustum <https://en.wikipedia.org/wiki/Viewing_frustum>`_ of the camera at image time (visualisation)

        Parameters
        ----------
        depth_min: float
            Near Viewing Plane
        depth_max: float
            Far Viewing Place

        Returns
        -------
        frustum: List[float]
        """
        return [

            # Near Viewing Plane
            spherical2cartesian(self._bounds.x0, self._bounds.y0, depth_min),
            spherical2cartesian(self._bounds.x0, self._bounds.y1, depth_min),
            spherical2cartesian(self._bounds.x1, self._bounds.y1, depth_min),
            spherical2cartesian(self._bounds.x1, self._bounds.y0, depth_min),

            # Far Viewing Plane
            spherical2cartesian(self._bounds.x0, self._bounds.y0, depth_max),
            spherical2cartesian(self._bounds.x0, self._bounds.y1, depth_max),
            spherical2cartesian(self._bounds.x1, self._bounds.y1, depth_max),
            spherical2cartesian(self._bounds.x1, self._bounds.y0, depth_max),
        ]

    def to_file(self, root):

        if not os.path.exists(os.path.dirname(root)):
            os.makedirs(os.path.dirname(root))

        # Save RGB Image
        Image.fromarray(self.image).save(os.path.join(root, "{}_rgb.png".format(self.hash)))

        # Save Depth Image
        np.save(os.path.join(root, "{}_depth.npy".format(self.hash)), self.depth)

        # Save Metadata
        with open(os.path.join(root, "{}_meta.json".format(self.hash)), 'w') as json_file:
            json.dump({
                "time": self.time,
                "bounds": self.bounds.dict()
            }, json_file)

    def __repr__(self):
        return "{}{}".format(self.__class__.__name__, self.image.shape)


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

    def __init__(self, name, confidence, representation, bounds, image):
        # type: (str, float, np.ndarray, Bounds, AbstractImage) -> None
        super(Face, self).__init__(name, confidence, bounds, image)

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


class ObjectObservations:
    """
    Object Observations for a particular Object Class
    """

    EPSILON = 0.2  # Distance in Metres within which observations are considered one single instance

    MIN_SAMPLES = 5  # Minimum number of observations for an instance
    MAX_SAMPLES = 50  # Maximum number of observations for an instance
    INSTANCE_TIMEOUT = 120  # Time in seconds without observation after which an instance no longer exists

    MIN_SAMPLES_MOVING = 3
    MAX_SAMPLES_MOVING = 8
    INSTANCE_TIMEOUT_MOVING = 30

    OBSERVATION_BOUNDS_AREA_THRESHOLD = 0.9  # If exceeded, observation is treated as a label for the scene instead
    OBSERVATION_TIMEOUT = 2  # Time in seconds for an observation to be considered 'recent'

    def __init__(self, name):
        # type: () -> None
        self._name = name
        self._moving = OBJECT_INFO[name]['moving'] if name in OBJECT_INFO else False

        self._min_samples = self.MIN_SAMPLES_MOVING if self._moving else self.MIN_SAMPLES
        self._max_samples = self.MAX_SAMPLES_MOVING if self._moving else self.MAX_SAMPLES
        self._instance_timeout = self.INSTANCE_TIMEOUT_MOVING if self._moving else self.INSTANCE_TIMEOUT

        self._observations = []
        self._instances = []

    @property
    def instances(self):
        # type: () -> List[Object]
        """
        Get individual object instances for this Object Class

        Returns
        -------
        instances: List[Object]
        """
        return self._instances

    def update_view(self, image):
        # type: (AbstractImage) -> None
        """
        Update Object Instances with Current Image

        Remove Observations one by one, when they are not longer where expected

        Parameters
        ----------
        image: AbstractImage
        """

        # If observation is a scene descriptor instead of an actual object, override clustering and use single instance
        if len(self._instances) == 1 and self._instances[0].image_bounds.area > self.OBSERVATION_BOUNDS_AREA_THRESHOLD:
            return

        # Limit observations & Instances to be within INSTANCE TIMEOUT
        self._observations = [obs for obs in self._observations if time() - obs.time < self._instance_timeout]
        self._instances = [ins for ins in self._instances if time() - ins.time < self._instance_timeout]

        # Go through observations oldest to newest
        for observation in self._observations[::-1]:

            # If observation could be done with current view
            if image.bounds.contains(observation.bounds.center):

                # Get Current Depth at Object Bounds to see if something might be occluding her view
                current_depth = image.get_depth(observation.image_bounds)
                current_depth = np.min(current_depth[current_depth != 0], initial=np.inf)

                # If nothing is occluding her view
                if current_depth > observation.depth - self.EPSILON:

                    # Check if recent observation of this object is made
                    found_recent_observation = False
                    for obs in self._observations:
                        if time() - obs.image.time > self.OBSERVATION_TIMEOUT:
                            break

                        if image.bounds.contains(obs.bounds.center):
                            found_recent_observation = True
                            break

                    # If no recent observation has been found -> remove one old observation
                    if not found_recent_observation:
                        self._observations.remove(observation)
                        break

    def add_observation(self, observation):
        """
        Add Observation of object with this Object Class

        Cluster Object Observations to figure out Object Instances

        Parameters
        ----------
        observation: Object
        """
        # If observation is a scene descriptor instead of an actual object, override clustering and use single instance
        try:
            observation_area = observation.image_bounds.area
        except:
            observation_area = 0

        if observation_area > self.OBSERVATION_BOUNDS_AREA_THRESHOLD:
            self._instances = [observation]
            return

        # Append Object Observation to all Observations
        self._observations.append(observation)

        # Get Positions of all Observations
        positions = [observation.position for observation in self._observations]

        instances = []
        removal = []

        # Cluster to find Object Instances
        cluster = DBSCAN(eps=self.EPSILON, min_samples=self._min_samples)
        cluster.fit(positions)

        unique_labels = np.unique(cluster.labels_)

        # Find newest instance per group add to Instances
        for label in unique_labels:

            group_indices = np.argwhere(cluster.labels_ == label).ravel()

            if label != -1:  # Skip Noisy Observations
                newest_instance = self._observations[group_indices[-1]]
                instances.append(newest_instance)

            removal.extend(group_indices[:-self._max_samples])

        self._instances = instances

        # Limit amount of stored observations
        self._observations = [self._observations[i] for i in range(len(self._observations)) if i not in removal]
