from astroquery.gaia import Gaia
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time


def query(query: str, *args, **kwargs):
    """
    Queries the gaia catalog, essentially just launches the job, catches the
    results and returns them

    Parameters:
        query (str): query string to send to gaia
    Returns:
        results (astropy.table.table.Table): job results
        job (astroquery.utils.tap.model.job.Job): job instance
    """
    job = Gaia.launch_job_async(query, *args, **kwargs)
    results = job.get_results()

    return results, job


def circles(list_coords: list[SkyCoord], radius: float, *args, **kwargs):
    """
    Queries the gaia catalog for all the targets in a circle of center `coords`
    and radius `radius`.

    Parameters:
        coords (list[astropy.coordinates.SkyCoord]): list of sky coordinates of
            circle centers
        radius (float): radius in degree or list of radiuses (length same as
            list_coords
    Returns:
        results (astropy.table.table.Table): job results
        job (astroquery.utils.tap.model.job.Job): job instance
    """
    if isinstance(radius, float):
        radius = [radius] * len(list_coords)
    assert len(list_coords) == len(
        radius
    ), "list_coords and radiuses have different dimensions"

    centers = []
    for coords in list_coords:
        coords_icrs = coords.transform_to("icrs")
        coords_icrs.location = None  # makes sure it's barycentric
        coords_icrs.obstime = Time("J2016.0")
        centers.append((coords_icrs.ra.deg, coords_icrs.dec.deg))

    where_clause = " OR ".join(
        [
            f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r})) = 1"
            for (ra, dec), r in zip(centers, radius)
        ]
    )
    query_str = "SELECT * FROM gaiadr3.gaia_source WHERE " + where_clause

    return query(query_str, *args, **kwargs)
