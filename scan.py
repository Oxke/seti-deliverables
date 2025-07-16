import numpy as np

# Extract RA and Dec from the dataframe
ra_sources = df_head["ra"].values
dec_sources = df_head["decl"].values


# Convert RA/Dec to 3D unit vectors for all sources
def radec_to_unitvec(ra_deg, dec_deg):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.vstack((x, y, z)).T


source_vecs = radec_to_unitvec(ra_sources, dec_sources)

# Define beam angular radius in degrees and convert to cosine threshold
beam_radius_deg = 8 / 60  # 8 arcmin
cos_radius = np.cos(np.radians(beam_radius_deg))

# Simulate a scan: grid of pointings
ra_scan = np.linspace(0, 360, 40)  # RA from 0 to 360 deg
dec_scan = np.linspace(-60, 60, 20)  # Dec from -60 to +60 deg
RA_grid, DEC_grid = np.meshgrid(ra_scan, dec_scan)
RA_grid_flat = RA_grid.ravel()
DEC_grid_flat = DEC_grid.ravel()

# Convert all pointing directions to unit vectors
pointing_vecs = radec_to_unitvec(RA_grid_flat, DEC_grid_flat)

# Compute number of sources in the beam for each pointing
counts = []
for pvec in pointing_vecs:
    dot_products = source_vecs @ pvec
    in_beam = dot_products >= cos_radius
    counts.append(np.count_nonzero(in_beam))

# Return RA, Dec grid and counts
scan_results = pd.DataFrame(
    {"ra": RA_grid_flat, "dec": DEC_grid_flat, "source_count": counts}
)
scan_results.head()
