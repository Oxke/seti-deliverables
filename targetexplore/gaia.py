from astroquery.gaia import Gaia
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time


def _make_query(where_clause, quality_cut, force_quality_cut):
    if quality_cut is False and not force_quality_cut:
        quality_cut = "1=1"
    if quality_cut and not force_quality_cut:
        query_head = """SELECT
    designation, ra, dec, parallax, parallax_error, phot_bp_mean_mag, phot_g_mean_mag, bp_rp,
    distance_gspphot, distance_gspphot_lower, distance_gspphot_upper,
    SQRT(
        POWER(1.0857 / phot_bp_mean_flux_over_error, 2) +
        POWER(1.0857 / phot_rp_mean_flux_over_error, 2)
    ) AS bp_rp_error,
    phot_g_mean_mag - 5 * LOG10(distance_gspphot) + 5 AS abs_g_mag,
    SQRT(
        POWER(1.0857 / phot_g_mean_flux_over_error, 2) +
        POWER(5.0 / (2.302585093 * distance_gspphot) *
        ((distance_gspphot_upper - distance_gspphot_lower) / 2), 2)
   ) AS abs_g_mag_error
FROM gaiadr3.gaia_source WHERE """
    else:
        query_head = "SELECT * FROM gaiadr3.gaia_source WHERE "
    query_str = f"{query_head} ({where_clause})"
    return query_str, quality_cut


def query(
    query: str, quality_cut=True, force_quality_cut=False, *args, **kwargs
):
    """
    Queries the gaia catalog, essentially just launches the job, catches the
    results and returns them

    Parameters:
        query (str): query string to send to gaia
        quality_cut (str|bool):
            - False: does not add anything to the query
            - True: adds the same quality cuts as in Czech's paper
            - (string): treated as additional query
    Returns:
        results (astropy.table.table.Table): job results
        job (astroquery.utils.tap.model.job.Job): job instance
    """
    if isinstance(quality_cut, str):
        query = f"{query} AND ({quality_cut})"
        if not force_quality_cut:
            query += """
AND distance_gspphot <> 0
AND phot_bp_mean_flux_over_error <> 0
AND phot_rp_mean_flux_over_error <> 0
AND phot_g_mean_flux_over_error <> 0
AND (distance_gspphot_upper - distance_gspphot_lower) <> 0
"""
    if quality_cut is True:
        query += """
AND distance_gspphot > 0
AND (distance_gspphot_upper - distance_gspphot_lower) > 0
AND parallax_over_error > 20
AND phot_g_mean_flux_over_error > 50
AND phot_rp_mean_flux_over_error > 20
AND phot_bp_mean_flux_over_error > 20
AND phot_bp_rp_excess_factor < 1.3 + 0.06 * POWER(phot_bp_mean_mag -
phot_rp_mean_mag, 2)
AND phot_bp_rp_excess_factor > 1.0 + 0.015 * POWER(phot_bp_mean_mag -
phot_rp_mean_mag, 2)
AND visibility_periods_used >= 8
AND astrometric_n_good_obs_al > 5
AND astrometric_chi2_al / (astrometric_n_good_obs_al - 5)
    < 1.44 * GREATEST(1, EXP(-0.4 * (phot_g_mean_mag - 19.5)))
"""
    try:
        job = Gaia.launch_job_async(query, *args, **kwargs)
    except Exception as e:
        return None, e, query
    results = job.get_results()

    return results, job, query


def _circles(
    list_coords: list[SkyCoord],
    radius: float,
    quality_cut=True,
    force_quality_cut=False,
    *args,
    **kwargs,
):
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

    coords_icrs = list_coords.transform_to("icrs")
    coords_icrs.location = None  # makes sure it's barycentric
    coords_icrs.obstime = Time("J2016.0")
    centers = np.array([list_coords.ra.deg, list_coords.dec.deg]).T

    where_clause = " OR ".join(
        [
            f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r})) = 1"
            for (ra, dec), r in zip(centers, radius)
        ]
    )

    return query(
        *_make_query(where_clause, quality_cut, force_quality_cut),
        force_quality_cut,
        *args,
        **kwargs,
    )


def _timed_circles(
    list_coords: list[SkyCoord],
    radius: float,
    quality_cut=True,
    force_quality_cut=False,
    *args,
    **kwargs,
):
    coords_icrs = list_coords.transform_to("icrs")
    coords_icrs.location = None  # makes sure it's barycentric
    coords_icrs.obstime = Time("J2016.0")
    centers = np.array([list_coords.ra.deg, list_coords.dec.deg]).T

    res_targets = []
    positive = ""
    negative = ""
    for (ra, dec), r in zip(centers, radius):
        negative = " or ".join([positive, negative])
        positive = f"contains(point('ICRS', ra, dec), circle('ICRL', {ra}, {dec}, {r})) = 1"
        where_clause = f"{positive} and not ({negative})"
        res_targets.append(
            query(
                *_make_query(where_clause, quality_cut, force_quality_cut),
                force_quality_cut,
                *args,
                **kwargs,
            )
        )

    return res_targets


def circles(*args, **kwargs):
    return _circles(*args, **kwargs)
    if mode == "one":
        return _circles(*args, **kwargs)
    if mode == "timed":
        return _timed_circles(*args, **kwargs)
