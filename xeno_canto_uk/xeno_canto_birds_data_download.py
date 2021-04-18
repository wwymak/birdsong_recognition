from ast import literal_eval
import pandas as pd
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    # name of 88 uk common bird species
    uk_bird_species = pd.read_csv('uk_common_birds_all.csv')
    uk_bird_species['genus']= uk_bird_species['genus'].apply(lambda x: x.strip())
    uk_bird_species['species'] = uk_bird_species['species'].apply(lambda x: x.strip())
    uk_bird_species['mergename'] = uk_bird_species.genus.str.cat(uk_bird_species.species, sep=' ')

    xeno_canto_eu = pd.read_csv('xeno-canto_europe_metadata_songs_ratingAB.csv')
    xeno_canto_eu['gen'] = xeno_canto_eu['gen'].apply(lambda x: x.strip())
    xeno_canto_eu['sp'] = xeno_canto_eu['sp'].apply(lambda x: x.strip())

    xeno_canto_eu['mergename'] = xeno_canto_eu.gen.str.cat(xeno_canto_eu.sp, sep=' ')
    xeno_canto_eu= xeno_canto_eu.merge(uk_bird_species, left_on='mergename', right_on='mergename')
    xeno_canto_eu.also = xeno_canto_eu.also.apply(literal_eval)
    xeno_canto_eu.also = xeno_canto_eu.also.apply(lambda x: np.nan if x==[''] else x)
    xeno_canto_eu = xeno_canto_eu[xeno_canto_eu.also.isna()]
    xeno_canto_eu['download_url'] = xeno_canto_eu.file.apply(lambda x: f"https:{x}")

    storage_path = Path('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu')
    errors = []
    for id, url in tqdm(zip(xeno_canto_eu['id'], xeno_canto_eu['download_url'])):
        r = requests.get(url)
        if r.status_code == 200:
            with open(storage_path/ f'{id}.mp3', 'wb') as f:
                f.write(r.content)    
        else:
            errors.append(id)
    pd.Series(errors).to_csv('xeno_canto_eu_errors.csv')