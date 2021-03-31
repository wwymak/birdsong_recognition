# to fetch metadata about recordings, you can use the xeno canto api
# https://www.xeno-canto.org/explore/api for query parameters
# https://www.xeno-canto.org/help/search for examples of how to specify search.
# here, we are filtering for birds recording in a bounding box around uk+ ireland
# if you want to filter for a certain rating, and songs only, add
# https://www.xeno-canto.org/api/2/recordings?query=box:49.951,-15.469,60.24,3.516+type:song+q_gt:C
import requests
import pandas as pd


if __name__ == '__main__':
    metadata = []
    session = requests.Session()
    def get_jobs():
        # xeno_canto_metadata_url = "https://www.xeno-canto.org/api/2/recordings?query=box:49.951,-15.469,60.24,3.516+type:song+q_gt:C"
        xeno_canto_metadata_url = "https://www.xeno-canto.org/api/2/recordings?query=area:europe+type:song+q_gt:C"
        first_page = session.get(xeno_canto_metadata_url).json()
        yield first_page
        num_pages = first_page['numPages']

        for page in range(2, num_pages + 1):
            next_page = session.get(xeno_canto_metadata_url, params={'page': page}).json()
            yield next_page

    for page in get_jobs():
        metadata.append(pd.DataFrame(page['recordings']))
    
    pd.concat(metadata).to_csv('xeno-canto_europe_metadata_songs_ratingAB.csv', index=False)