List of data files:

* **Instagram_userday_terms_raw.zip**
* **Flickr_userday_terms_raw.zip** Distinct terms grouped by distinct userday.

  **Contents:**
  * flickr_sunrise_terms_geotagged_grouped.csv
  * flickr_sunset_terms_geotagged_grouped.csv
  * instagram_sunrise_terms_geotagged_grouped.csv
  * instagram_sunrise_terms_nonegeotagged_grouped.csv
  * instagram_sunset_terms_geotagged_grouped.csv
  * instagram_sunset_terms_nonegeotagged_grouped.csv

  **Structure:**

  | userday | userday_terms | userday_season_id | | ------------ | --------------------------------------- | ----------------- | | userday-hash | "{term1,term2,term3,term4,term5,term6}" | 2 |

  userday_season_id: reference to season Note: Classification is based on Meteorological seasons from Location and Date, but using the static range from the temperate zone.

  | Season | ID | | ------------------ | ---- | | Multiple/Ambiguous | 0 | | Northern spring | 1 | | Northern summer | 2 | | Northern fall | 3 | | Northern winter | 4 | | Southern spring | -1 | | Southern summer | -2 | | Southern fall | -3 | | Southern winter | -4 |

  See [2020-11-30_Create_UserdayTerm_Extracts_RAW.sql][2020-11-30_Create_UserdayTerm_Extracts_RAW.sql][]
* **2020-12-07_FlickrInstagram_random1M.zip** Distinct terms grouped by distinct userday, but for a random selection of 1 Million userdays
  * from \~200 Million Instagram posts
  * from \~350 Million Flickr posts

  See [2020-12-03_random_sample_tfidf.sql][2020-12-03_random_sample_tfidf.sql]
* **Flickr_userday_location_ref.zip**
* **Instagram_userday_locations_ref.zip** Location-reference for Distinct Userdays
  * Locations from all posts of a distinct user day grouped
  * Centroid of Boundary
  * Centroid Intersection of Boundary in 100km Grid

  | userday | xbin | ybin | su_a3 | | ------------------------------------------- | -------- | -------- | ----- | | +++7YqS8pYaZG1GskMdjVpMWme1VvgXEJnHlVobsz64 | 11459904 | -920048 | IDN | | +++ZxQ4pLBONKl1k9DCQR9hyArv0esztYlFXsQd3/AY | -4240096 | -2420048 | BRA |
* xbin/ybin: Index reference to 100km grid, see [Jupyter Notebook](https://ad.vgiscience.org/sunset-sunrise-paper/05_countries.html)
* country reference (from grid intersection), based on [Natural Earth map units shapefile (1:50m)](https://www.naturalearthdata.com/downloads/50m-cultural-vectors/50m-admin-0-details/) - Column: `SU_A3`

See [2020-12-07_location_userday_extracts_instagram.sql][2020-12-07_location_userday_extracts_instagram.sql]