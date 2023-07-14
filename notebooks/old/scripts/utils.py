import requests
import re
import numpy as np
import json

def get_identifier_type(identifier):
    identifier = str(identifier)
    if 'doi' in identifier:
        return 'doi'
    elif re.match(r'\d+', identifier):
        return 'pmid'
    elif 'http' in identifier:
        return 'other_url'
    else:
        return None



def id_convert(ids):
    service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=distribution-nm&email=javier.millanacosta@maastrichtuniversity.nl&ids={}&format=json"
    USER_AGENT = "Mozilla/5.0"
    url = service_root.format(ids)
    response = requests.get(url, headers={"User-Agent": USER_AGENT})
    response_code = response.status_code
    try:
        if response_code < 200 or response_code >= 300:
            #print(f"Non-200 response code for {url}")
            return False
        else:
            
            if 'pmcid' in response.json()['records'][0].keys():
                
                pmcid = response.json()['records'][0]['pmcid']
                return pmcid
            else:
                return ""
                
    except Exception as e:
        #print(f"An error ({str(e)}) occurred for {id}")
        return False


def convert_to_pmcid(row):
    if row['identifier_type'] in ['pmid', 'doi']:
        return id_convert(row['provided_identifier'])
    else:
        return None
    
def id_set(df):
    seen = []
    for index, row in df.iterrows():
        id = row['provided_identifier']
        if id not in seen:
            seen.append(id)
            converted = id_convert(id)
            if converted != False:
                pmcid = converted[0]
                doi = converted[1]
                df.at[index, 'pmcid'] = pmcid
                df.at[index, 'doi'] = doi
    return df

def get_full_text(pmcid, doi, elsevier_api_key, filepath='../data/docs/'):
    # Europe PMC and Elsevier URLs
    url_pmc = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    url_elsevier = f'https://api.elsevier.com/content/article/doi/{doi}?APIKey={elsevier_api_key}&httpAccept=text/XML'
    USER_AGENT = "Mozilla/5.0"
    url_willey = f'https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}?download=true'


    # Try Europe PMC first
    response = requests.get(url_pmc, headers={"User-Agent": USER_AGENT})
    if response.status_code == 200:
        response_text = response.text
        filepath = filepath + doi.split("/")[-1] + '.xml'
        #print(f'\t{doi} full text XML retrieved from Europe PMC to {filepath}')
        with open(filepath, 'w') as f:
            f.write(response_text)
        return filepath

    # If not found in Europe PMC, try Elsevier
    response = requests.get(url_elsevier, headers={"User-Agent": USER_AGENT})
    if response.status_code == 200:
        response_text = response.text
        filepath = filepath + doi.split("/")[-1] + '.xml'
        #print(f'{doi} full text XML retrieved from Elsevier to {filepath}')
        with open(filepath, 'w') as f:
            f.write(response_text)
        return filepath
    

    # Paper not found in both sources
    error_msg = f"\t|\n\t -> https://doi.org/{doi} needs to be added manually to ../data/docs"
    #print(error_msg)
    return ""


