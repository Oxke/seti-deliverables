import numpy as np
from astropy.coordinates import SkyCoord
from targetexplore.gaia import circle


class Targets:
    """
    Represents a set of targets. Main inner class of the module

    ...

    Attributes
    ----------
    centers: list[SkyCoord]
    radius: radius of list of radiuses for observations
    targets: table of targets (after self.query(), otherwise None)

    Methods
    -------

    info(additional=""): Prints the person's name and age.
    """
    def __init__(self, centers: list[SkyCoord], radius, query=True):
        """
        Initialize a Targets object
        
        Parameters:
            centers (list of Skycoord): SkyCoord coordinates of centers to point
            radius: either a single radius or a list, in arcminutes
            query (boolean): if true, immediately queries Gaia for the targets
        """

        self.centers = np.array(centers)
        self.radius = np.array(radius)
        self.targets = None

        if query: self.query()

    def query(self, *args, **kwargs):
        self.targets, job = circle(self.centers, self.radius / 60, *args, **kwargs) # degrees
        return job

    
