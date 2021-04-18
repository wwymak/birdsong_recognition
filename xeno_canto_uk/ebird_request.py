import requests
import pandas as pd
# fill in api key you get from browser console
API_KEY = ""

def get_url(bird_name):
    url = f"https://api.ebird.org/v2/ref/taxon/find?locale=en_UK&cat=species&key={API_KEY}&q={bird_name}"
    r = requests.get(url)
    return r.json()

if __name__=="__main__":
    df = pd.read_csv('uk_common_birds.csv')
    for row in df[df.ebird_code.isna()].iterrows():
        bird_name = f"{row[1].genus.strip()} {row[1].species.strip()}"
        data = get_url(bird_name)
        if len(data) == 1:
            df.loc[row[0]]['ebird_code'] = data[0]['code']
            
    df.to_csv('uk_common_birds_all.csv', index=False)