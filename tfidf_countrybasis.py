# import dependencies
import math
import pickle
import functools
import numpy as np
import pandas as pd
import sklearn.metrics
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count

'''
THIS IS A VARIATION TO THE tfidf.py SCRIPT. HERE TF-IDF IS CALCULATED BASED ON DOCUMENT FREQUENCY FROM 5M RANDOM SAMPLES 
FOR EACH COUNTRY. TERM FREQUENCY IS STILL BASED ON SUNSET/SUNRISE POSTS ALSO PER COUNTRY

ANACONDA ENV: geoenv
PROCESSING STEPS (use Counter class!)
1. calculate document frequency for all terms(sunset or sunrise) (flickr or Instagram) terms
2. calculate term frequency for every country
3. calculate tf-idf for each country and store the top X in csv (line: country,top_term_1,top_term_2,...)
--> tf-idf(t,d) = tf(t,d) * log(N/(df + 1))
t — term (word)
d — document (set of words)
N — count of corpus
Source: https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
'''
# define if term frequency is calculated using all available cores
MULTIPROCESSING = False
# document frequency is based on random flickr set instead of only either sunset or sunrise data
BASE_DOCUMENT_FREQUENCY_RANDOM = True
RANDOM_LOCATION_PRESENT = True
random_term_col = 'userday_terms'# due to error of Alex in exporting some mismatch, can be adjusted here
# min. amount of unique userposts a country must have to be included in the processing
THRESHOLD = 25
# if location (country code) is already present (true) in flickr dataset the merge with a location dataset can be skipped
LOCATION_PRESENT = True
if LOCATION_PRESENT:
    # define search col where the flickr terms are found
    term_col = 'user_terms'
else:
    term_col = 'userday_terms'
# define which phenomenon is analysed
MODE = 'SUNSET' #OR SUNRISE
SOURCE = 'INSTAGRAM' # OR INSTAGRAM OR ALL
TOP_TERMS = 20 # the amount of terms with the highest tf-idf values per country that will included

if SOURCE == 'FLICKR':
    # random sample is used to build the entire vocabulary for tf-idf. columns: userday, userday_terms, userday_season_id
    RANDOM_PATH = Path("./Semantic_analysis/2021-02-02_InstagramFlickr_random5m_userterms_country/flickr_random5m_userterms_countries_grouped.csv")
    RANDOM_LOCATIONREF_PATH = Path("./Semantic_analysis(2020-12-07_FlickrInstagram_random1M/??????????????????????????????.csv")
    if MODE == 'SUNSET':
        # spatial reference to unique userday hashaes. columns: userday, xbin, ybin, su_a3 (country code)
        LOCATIONREF_PATH = Path("./Semantic_analysis/Flickr_userday_location_ref/flickr_sunset_userday_gridloc.csv")
        # actual sunset/sunrise data. columns: userday, userday_terms, userday_season_id (Multiple/Ambiguous 0, Northern spring 1, Northern summer 2, Northern fall 3, Northern winter 4, Southern spring -1, Southern summer -2, Southern fall -3, Southern winter -4
        DATA_PATH = Path("./Semantic_analysis/2021-01-28_country_userterms/flickr_sunset_terms_user_country.csv") # CHANGE HERE IF NECESSARY
        # OUTPUT store path
        TF_IDF_STORE_PATH = Path("./Semantic_analysis/20210204_FLICKR_SUNSET_random_country_tf_idf.csv") # CHANGE HERE IF NECESSARY

    elif MODE == 'SUNRISE':
        # spatial reference to unique userday hashaes. columns: userday, xbin, ybin, su_a3 (country code)
        LOCATIONREF_PATH = Path("./Semantic_analysis/Flickr_userday_location_ref/flickr_sunrise_userday_gridloc.csv")
        # actual sunset/sunrise data. columns: userday, userday_terms, userday_season_id (Multiple/Ambiguous 0, Northern spring 1, Northern summer 2, Northern fall 3, Northern winter 4, Southern spring -1, Southern summer -2, Southern fall -3, Southern winter -4
        DATA_PATH = Path("./Semantic_analysis/Flickr_userday_terms_raw/flickr_sunrise_terms_geotagged_grouped.csv")
        # OUTPUT store path
        TF_IDF_STORE_PATH = Path("./Semantic_analysis/20210204_FLICKR_SUNRISE_country_tf_idf.csv")

elif SOURCE == 'INSTAGRAM':
    # random sample is used to build the entire vocabulary for tf-idf. columns: userday, userday_terms, userday_season_id
    RANDOM_PATH = Path("./Semantic_analysis/2021-02-02_InstagramFlickr_random5m_userterms_country/instagram_random5m_userterms_countries_grouped.csv")
    if MODE == 'SUNSET':
        # spatial reference to unique userday hashaes. columns: userday, xbin, ybin, su_a3 (country code)
        LOCATIONREF_PATH = Path("./Semantic_analysis/Flickr_userday_location_ref/flickr_sunset_userday_gridloc.csv")
        # actual sunset/sunrise data. columns: userday, userday_terms, userday_season_id (Multiple/Ambiguous 0, Northern spring 1, Northern summer 2, Northern fall 3, Northern winter 4, Southern spring -1, Southern summer -2, Southern fall -3, Southern winter -4
        DATA_PATH = Path("./Semantic_analysis/2021-01-28_country_userterms/instagram_sunset_terms_user_country.csv")  # CHANGE HERE IF NECESSARY
        # OUTPUT store path
        TF_IDF_STORE_PATH = Path("./Semantic_analysis/20210204_INSTAGRAM_SUNSET_random_country_tf_idf.csv")  # CHANGE HERE IF NECESSARY

    elif MODE == 'SUNRISE':
        # spatial reference to unique userday hashaes. columns: userday, xbin, ybin, su_a3 (country code)
        LOCATIONREF_PATH = Path("./Semantic_analysis/Flickr_userday_location_ref/flickr_sunrise_userday_gridloc.csv")
        # actual sunset/sunrise data. columns: userday, userday_terms, userday_season_id (Multiple/Ambiguous 0, Northern spring 1, Northern summer 2, Northern fall 3, Northern winter 4, Southern spring -1, Southern summer -2, Southern fall -3, Southern winter -4
        DATA_PATH = Path("./Semantic_analysis/2021-01-28_country_userterms/instagram_sunrise_terms_user_country.csv")
        # OUTPUT store path
        TF_IDF_STORE_PATH = Path("./Semantic_analysis/20210204_INSTAGRAM_SUNRISE_country_tf_idf.csv")
    
def load_data():
    print('loading data...')
    if BASE_DOCUMENT_FREQUENCY_RANDOM:
        # load random flickr dataset which forms the document corpus
        random_w_locationref_df = pd.read_csv(RANDOM_PATH, encoding='utf-8')
        # 1.2 retrieve unique country codes for iteration
        country_codes_random = random_w_locationref_df['su_a3'].unique()
        # remove nan country codes from array (issues with further processing) - index value 17
        ###country_codes = np.delete(country_codes, 17)

        # load the sunset/sunrise dataset which forms the term corpus
        data_df = pd.read_csv(DATA_PATH, encoding='utf-8')
        if not LOCATION_PRESENT:
            locationref_df = pd.read_csv(LOCATIONREF_PATH)
            # merge dataframes basde on userday hash
            data_w_locationref_df = data_df.merge(locationref_df, how='left', on='userday')
        else:
            data_w_locationref_df = data_df
        # 1.2 retrieve unique country codes for iteration
        country_codes_sun = data_w_locationref_df['su_a3'].unique()
        # remove nan country codes from array (issues with further processing) - index value 17
        # if not LOCATION_PRESENT:
        #    country_codes = np.delete(country_codes_all, 17)

        # merge country codes and only keep the ones present in both datasets
        country_codes = []
        [country_codes.append(code)if code in country_codes_sun else print(f'country code {code} not in both datasets') for code in country_codes_random]

        return data_w_locationref_df, random_w_locationref_df, country_codes

def text_processing(document_vocabulary):
    '''
    - remove stopwords
    - check for numbers
    - returned cleaned term list
    '''

    return document_vocabulary

def calc_document_frequency(data_random_w_locationref_df, country_codes):
    '''
    PROCESSING STEP 1
    1. create new column with set of terms per post
    2. calculate Counter object over all posts to get document frequency for all terms
    '''
    print('calculating document frequency...')
    data_random_w_locationref_df[random_term_col] = data_random_w_locationref_df[random_term_col].apply(lambda x: x.strip('{}'))
    # create set of all term lists
    data_random_w_locationref_df['userday_terms_set'] = data_random_w_locationref_df[random_term_col].apply(lambda x: list(set(x.split(','))))
    # !preprocessing for term frequency: add all userday terms to one big list containing also duplicates and not a unique set like for the document frequency
    data_random_w_locationref_df['userday_terms_list'] = data_random_w_locationref_df[random_term_col].apply(lambda x: list(x.split(',')))
    # drop the userday_terms column from dataframe
    data_random_w_locationref_df.drop([random_term_col], axis=1)
    # create dictionary that stores the country specific term frequencies
    country_document_frequency_dict = {}
    # 2. iterate over all country codes and merge Counter objects
    for index, country_code in enumerate(country_codes, 1):
        # create sub df that only contains random posts of given country code
        country_code_random_df = data_random_w_locationref_df[data_random_w_locationref_df['su_a3'] == country_code]
        # merge all set lists and create a Counter object on it
        document_vocabulary = country_code_random_df['userday_terms_set'].explode()
        # clean out text
        cleaned_document_vocabulary = text_processing(document_vocabulary)
        # create Counter object --> actual document frequency of every term
        country_code_term_frequency = Counter(cleaned_document_vocabulary)
        # store country specific document frequency in corresponding dictionary
        country_document_frequency_dict[country_code] = country_code_term_frequency

    return country_document_frequency_dict, data_random_w_locationref_df


def calc_term_frequency(data_w_locationref_df, country_codes):
    '''
    1. calculate Counter object for each post (not on set term list like for document frequency)
    2. merge Counter objects for posts of the same country
    '''
    data_w_locationref_df[term_col] = data_w_locationref_df[term_col].apply(lambda x: x.strip('{}'))
    # create set of all term lists
    data_w_locationref_df['userday_terms_set'] = data_w_locationref_df[term_col].apply(lambda x: list(set(x.split(','))))
    # !preprocessing for term frequency: add all userday terms to one big list containing also duplicates and not a unique set like for the document frequency
    data_w_locationref_df['userday_terms_list'] = data_w_locationref_df[term_col].apply(lambda x: list(x.split(',')))
    # drop the userday_terms column from dataframe
    data_w_locationref_df.drop([term_col], axis=1)
    # create dictionary that stores the country specific term frequencies
    country_term_frequency_dict = {}
    # 2. iterate over all country codes and merge Counter objects
    for index, country_code in enumerate(country_codes, 1):
        # create sub df that only contains posts of given country code
        country_code_df = data_w_locationref_df[data_w_locationref_df['su_a3'] == country_code]
        if len(country_code_df.index.values) >= THRESHOLD:
            # merge all set lists and create a Counter object on it
            country_code_term_vocabulary = country_code_df['userday_terms_list'].explode()
            # create Counter object --> actual document frequency of every term
            country_code_term_frequency = Counter(country_code_term_vocabulary)
            country_term_frequency_dict[country_code] = country_code_term_frequency
        else:
            print(f'EXCLUDED {country_code}')
    return country_term_frequency_dict


def calc_tf_idf(data_random_w_locationref_df, country_document_frequency_dict, country_term_frequency_dict):
    '''
    1. iterate over each dict containing term frequencies for every country
    2. find the document frequency of that term in the document_frequency_dict
    3. caluclate tf-idf for that term and that country and store in a tuple list
    4. rank
    5. save output

    --> tf-idf(t,d) = tf(t,d) * log(N/(df + 1))
    t — term (word)
    d — document (set of words)
    N — count of corpus
    '''
    print('calculating tf-idf...')
    # dictionary that stores all tf-idf's per country
    tf_idf_country_dict = {}
    # track skipped terms
    skipped_term_list = []
    dict_len = len(country_term_frequency_dict.items())
    for index, (country_code, country_term_frequencies) in enumerate(country_term_frequency_dict.items(), 1):
        print(f'tf-idf for country {country_code} - {index} of {dict_len}')
        # get N (the amount of Flickr posts)
        N = data_random_w_locationref_df[data_random_w_locationref_df['su_a3'] == country_code].shape[0]
        # create new list for that country_code that holds the tf-idf values for each term
        tf_idf_list = []
        # iterate over Counter object
        for term, tf in country_term_frequencies.items():
            # get document frequency for the same term (and check if it is present)
            try:
                df = country_document_frequency_dict[country_code][term]
            except Exception as e:
                # print(f'error: {e}')
                # print(f'no document frequency found for term {term} - SKIPPING')
                skipped_term_list.append(term)
                continue
            # calculate tf-idf for this term and round it to two digits
            tf_idf = round(tf * math.log((N/(df + 1)), 10), 2)
            tf_idf_list.append((term, tf_idf))
        # sort the list based on tf-idf value in descending order
        tf_idf_list.sort(key=lambda tup: tup[1], reverse=True)
        # store list with country code as key in tf-idf dic
        tf_idf_country_dict[country_code] = tf_idf_list
        print(f'skipped {len(skipped_term_list)} terms')

    return tf_idf_country_dict

def save_output(tf_idf_country_dict, TOP_TERMS=20):
    '''
    1. iterate over country in dict
    2. iterate over sorted list containing tuples (term, tf-idf value)
    3. write string line for each country countaining the top X terms followed by their tf-idf scores
    '''
    print('saving output...')
    # write output file header
    header_line = 'COUNTRY_CODE'
    for i in range(TOP_TERMS):
        header_line += f',TERM_{(i+1)},TF_IDF_{(i+1)}'
    header_line += '\n'
    with open(TF_IDF_STORE_PATH, 'at', encoding='utf-8') as f:
        f.write(header_line)
    for country_code, tf_idf_list in tf_idf_country_dict.items():
        # check if enough terms are actually present in the tf_idf_list
        try:
            top_tf_idf_list = tf_idf_list[:TOP_TERMS]
        except Exception as e:
            print(f'{e}: less than TOP_terms given')
            top_tf_idf_list = top_tf_idf_list
        # iterate trough tuple list and create output line
        line = f'{country_code}'
        for tf_idf_tuple in top_tf_idf_list:
            # term, tf-idf value
            line += f',{tf_idf_tuple[0]},{tf_idf_tuple[1]}'
        line += '\n'
        with open(TF_IDF_STORE_PATH, 'at', encoding='utf-8') as f:
            f.write(line)
    print('done.')


if __name__ == '__main__':
    data_w_locationref_df, data_random_w_locationref_df, country_codes = load_data()
    country_document_frequency_dict, data_random_w_locationref_df = calc_document_frequency(data_random_w_locationref_df, country_codes)
    country_term_frequency_dict = calc_term_frequency(data_w_locationref_df, country_codes)
    tf_idf_country_dict = calc_tf_idf(data_random_w_locationref_df, country_document_frequency_dict, country_term_frequency_dict)
    save_output(tf_idf_country_dict, TOP_TERMS)