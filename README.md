# FluxNet

| Field    | Description                                                                           |
| -------- | ------------------------------------------------------------------------------------- |
| `time`   | 1979–2021 (yearly data)                                                               |
| `x`, `y` | Already in Antarctic Polar Stereographic Projection [EPSG:3031](https://epsg.io/3031) |
| `smb`    | **mm i.e. per year** (i.e. = ice equivalent)                                                                 |

# Inputs

## Ice thickenss (h)

We use the Bedmap collection of ice thickness measurements. We combine all standardised .csv files from the Bedmap1, Bedmap2 and Bedmap3 collections from the [UK Polar Data Centre](https://www.bas.ac.uk/data/uk-pdc/). The lists of .csv files are visible on [this Bristish Antarctic Survey (BAS) webpage](https://www.bas.ac.uk/project/bedmap/#data).

Bedmap(3) references:
- *Pritchard, Hamish D., et al. "Bedmap3 updated ice bed, surface and thickness gridded datasets for Antarctica." Scientific data 12.1 (2025): 414.*
- *Frémand, Alice C., et al. "Antarctic Bedmap data: Findable, Accessible, Interoperable, and Reusable (FAIR) sharing of 60 years of ice bed, surface, and thickness data." Earth System Science Data 15.7 (2023): 2695-2710.*

![Ice thickess observations from Bedmap 1+2+3](figures/ice_thickness_points_onshelf.png)

## Ice velocity (v)

We use [MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1](https://nsidc.org/data/nsidc-0754/versions/1)

Reference:
- *Mouginot, J., Rignot, E. & Scheuchl, B. (2019). MEaSUREs Phase-Based Antarctica Ice Velocity Map. (NSIDC-0754, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PZ3NJ5RXRH10. Date Accessed 10-02-2025.*

![Ice velocity (phase-based)](figures/ice_velocity_phase_oshelf.png)

## Surface mass balance (smb)

We use interpolated [Higher Antarctic ice sheet accumulation and surface melt rates revealed at 2 km resolution](https://zenodo.org/records/10007855)
(filename: smb_rec.1979-2021.BN_RACMO2.3p2_ANT27_ERA5-3h.AIS.2km.YY.nc:)

Reference:
- *Noël, Brice, et al. "Higher Antarctic ice sheet accumulation and surface melt rates revealed at 2 km resolution." Nature communications 14.1 (2023): 7949.*

## Thickening rates

[ATLAS/ICESat-2 L3B Gridded Antarctic and Arctic Land Ice Height Change, Version 4]
(https://nsidc.org/data/atl15/versions/4)

Use command line: https://search.earthdata.nasa.gov/downloads/5132364764

# Reference basal mass balance map

- **Antarctic iceshelf melt rates**: Average basal melt rates for Antarctic ice shelves for the 2010–2018 period at high spatial resolution, estimated using CryoSat-2 data. This data file was last updated on 2020-06-11.


[](https://library.ucsd.edu/dc/object/bb0448974g)

Download to local with   
``wget https://library.ucsd.edu/dc/object/bb0448974g/_3_1.h5/download``

Reference:
- *Adusumilli, Susheel, et al. "Interannual variations in meltwater input to the Southern Ocean from Antarctic ice shelves." Nature geoscience 13.9 (2020): 616-620.* [Link to paper on Nature.](https://www.nature.com/articles/s41561-020-0616-z)
