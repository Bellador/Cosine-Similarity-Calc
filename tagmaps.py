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
BASE_DOCUMENT_FREQUENCY_RANDOM = False
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
SOURCE = 'FLICKR' # OR INSTAGRAM OR ALL
TOP_TERMS = 20 # the amount of terms with the highest tf-idf values per country that will included

if SOURCE == 'FLICKR':
    # random sample is used to build the entire vocabulary for tf-idf. columns: userday, userday_terms, userday_season_id
    FLICKR_RANDOM1M_PATH = Path("./Semantic_analysis(2020-12-07_FlickrInstagram_random1M/flickr_random1m_terms_geotagged_grouped.csv")
    FLICKR_RANDOM1M_LOCATIONREF_PATH = Path("./Semantic_analysis(2020-12-07_FlickrInstagram_random1M/??????????????????????????????.csv")
    if MODE == 'SUNSET':
        # spatial reference to unique userday hashaes. columns: userday, xbin, ybin, su_a3 (country code)
        FLICKR_LOCATIONREF_PATH = Path("./Semantic_analysis/Flickr_userday_location_ref/flickr_sunset_userday_gridloc.csv")
        # actual sunset/sunrise data. columns: userday, userday_terms, userday_season_id (Multiple/Ambiguous 0, Northern spring 1, Northern summer 2, Northern fall 3, Northern winter 4, Southern spring -1, Southern summer -2, Southern fall -3, Southern winter -4
        FLICKR_PATH = Path("./Semantic_analysis/2021-01-28_country_userterms/flickr_sunset_terms_user_country.csv") # CHANGE HERE IF NECESSARY
        # OUTPUT store path
        TF_IDF_FLICKR_STORE_PATH = Path("./Semantic_analysis/20210201_FLICKR_SUNSET_country_tf_idf.csv") # CHANGE HERE IF NECESSARY

    elif MODE == 'SUNRISE':
        # spatial reference to unique userday hashaes. columns: userday, xbin, ybin, su_a3 (country code)
        FLICKR_LOCATIONREF_PATH = Path("./Semantic_analysis/Flickr_userday_location_ref/flickr_sunrise_userday_gridloc.csv")
        # actual sunset/sunrise data. columns: userday, userday_terms, userday_season_id (Multiple/Ambiguous 0, Northern spring 1, Northern summer 2, Northern fall 3, Northern winter 4, Southern spring -1, Southern summer -2, Southern fall -3, Southern winter -4
        FLICKR_PATH = Path("./Semantic_analysis/Flickr_userday_terms_raw/flickr_sunrise_terms_geotagged_grouped.csv")
        # OUTPUT store path
        TF_IDF_FLICKR_STORE_PATH = Path("./Semantic_analysis/FLICKR_SUNRISE_country_tf_idf.csv")

elif SOURCE == 'INSTAGRAM':
    INSTAGRAM_RANDOM1M_PATH = Path("./Semantic_analysis/2020-12-07_FlickrInstagram_random1M/instagram_random1m_terms_geotagged_grouped.csv")

def load_data():
    print('loading data...')
    if BASE_DOCUMENT_FREQUENCY_RANDOM:
        flickr_random_1m_df = pd.read_csv(FLICKR_RANDOM1M_PATH, encoding='utf-8')
        flickr_random_1m_locationref_df = pd.read_csv(FLICKR_RANDOM1M_LOCATIONREF_PATH, encoding='utf-8')
        # merge dataframes basde on userday hash
        flickr_random_1m_w_locationref_df = flickr_random_1m_df.merge(flickr_random_1m_locationref_df, how='left', on='userday')
        # 1.2 retrieve unique country codes for iteration
        country_codes = flickr_random_1m_w_locationref_df['su_a3'].unique()
        # remove nan country codes from array (issues with further processing) - index value 17
        ###country_codes = np.delete(country_codes, 17)
        return flickr_random_1m_w_locationref_df, country_codes
    else:
        flickr_df = pd.read_csv(FLICKR_PATH, encoding='utf-8')
        if not LOCATION_PRESENT:
            flickr_locationref_df = pd.read_csv(FLICKR_LOCATIONREF_PATH)
            # merge dataframes basde on userday hash
            flickr_w_locationref_df = flickr_df.merge(flickr_locationref_df, how='left', on='userday')
        else:
            flickr_w_locationref_df = flickr_df
        # 1.2 retrieve unique country codes for iteration
        country_codes = flickr_w_locationref_df['su_a3'].unique()
        # remove nan country codes from array (issues with further processing) - index value 17
        if not LOCATION_PRESENT:
            country_codes = np.delete(country_codes, 17)

        return flickr_w_locationref_df, country_codes

def text_processing(document_vocabulary):
    '''
    - remove stopwords
    - check for numbers
    - returned cleaned term list
    '''
    print('text processing...')
    return document_vocabulary

def calc_document_frequency(flickr_w_locationref_df):
    '''
    PROCESSING STEP 1
    1. create new column with set of terms per post
    2. calculate Counter object over all posts to get document frequency for all terms
    '''
    print('calculating document frequency...')
    flickr_w_locationref_df[term_col] = flickr_w_locationref_df[term_col].apply(lambda x: x.strip('{}'))
    # create set of all term lists
    flickr_w_locationref_df['userday_terms_set'] = flickr_w_locationref_df[term_col].apply(lambda x: list(set(x.split(','))))
    # !preprocessing for term frequency: add all userday terms to one big list containing also duplicates and not a unique set like for the document frequency
    flickr_w_locationref_df['userday_terms_list'] = flickr_w_locationref_df[term_col].apply(lambda x: list(x.split(',')))
    # drop the userday_terms column from dataframe
    flickr_w_locationref_df.drop([term_col], axis=1)
    # merge all set lists and create a Counter object on it
    document_vocabulary = flickr_w_locationref_df['userday_terms_set'].explode()
    # clean out text
    cleaned_document_vocabulary = text_processing(document_vocabulary)
    # create Counter object --> actual document frequency of every term
    document_frequency = Counter(cleaned_document_vocabulary)
    return document_frequency, flickr_w_locationref_df

def calc_document_frequency_random_posts_per_country(flickr_w_locationref_df):
    '''
    This version of the document frequency calculation is based on random flickr posts compared to only all sunset or sunrise flickr posts
    We saw that toponyms were high up in the tf-idf values in the previous approach since the same did not appear really oft across many different sunset/sunrise posts
    Given random flickr posts from all countries specifically will increase the document frequency of toponyms and therefore lower their final tf-idf score which is what we want to achieve
    PROCESSING STEP 1
    1. create new column with set of terms per post
    2. calculate Counter object over all posts to get document frequency for all terms
    '''
    print('calculating document frequency...')
    flickr_w_locationref_df[term_col] = flickr_w_locationref_df[term_col].apply(lambda x: x.strip('{}'))
    # create set of all term lists
    flickr_w_locationref_df['userday_terms_set'] = flickr_w_locationref_df[term_col].apply(lambda x: list(set(x.split(','))))
    # !preprocessing for term frequency: add all userday terms to one big list containing also duplicates and not a unique set like for the document frequency
    flickr_w_locationref_df['userday_terms_list'] = flickr_w_locationref_df[term_col].apply(lambda x: list(x.split(',')))
    # drop the userday_terms column from dataframe
    flickr_w_locationref_df.drop([term_col], axis=1)
    # merge all set lists and create a Counter object on it
    document_vocabulary = flickr_w_locationref_df['userday_terms_set'].explode()
    # clean out text
    cleaned_document_vocabulary = text_processing(document_vocabulary)
    # create Counter object --> actual document frequency of every term
    document_frequency = Counter(cleaned_document_vocabulary)
    return document_frequency, flickr_w_locationref_df

def calc_term_frequency(flickr_w_locationref_df, country_codes, tracker=1):
    '''
    1. calculate Counter object for each post (not on set term list like for document frequency)
    2. merge Counter objects for posts of the same country
    '''
    # create dictionary that stores the country specific term frequencies
    country_term_frequency_dict = {}
    # 2. iterate over all country codes and merge Counter objects
    for index, country_code in enumerate(country_codes, 1):
        # # for testing
        # if index == 3:
        #     break
        # create sub df that only contains posts of given country code
        country_code_df = flickr_w_locationref_df[flickr_w_locationref_df['su_a3'] == country_code]
        if len(country_code_df.index.values) >= THRESHOLD:
            print(f'Process {tracker} - calculating term frequency for {country_code}:   {index} of {len(country_codes)}')
            # merge all set lists and create a Counter object on it
            country_code_term_vocabulary = country_code_df['userday_terms_list'].explode()
            # create Counter object --> actual document frequency of every term
            country_code_term_frequency = Counter(country_code_term_vocabulary)
            country_term_frequency_dict[country_code] = country_code_term_frequency
        else:
            print(f'Process {tracker} - EXCLUDED {country_code}')
    return country_term_frequency_dict


def calc_tf_idf(flickr_w_locationref_df, document_frequency, country_term_frequency_dict):
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
    # get N (the amount of Flickr posts)
    N = flickr_w_locationref_df.shape[0]
    dict_len = len(country_term_frequency_dict.items())
    for index, (country_code, country_term_frequencies) in enumerate(country_term_frequency_dict.items(), 1):
        print(f'tf-idf for country {country_code} - {index} of {dict_len}')
        # create new list for that country_code that holds the tf-idf values for each term
        tf_idf_list = []
        # iterate over Counter object
        for term, count in country_term_frequencies.items():
            df = document_frequency[term]
            # calculate tf-idf for this term and round it to two digits
            tf_idf = round(count * math.log((N/(df + 1)), 10), 2)
            tf_idf_list.append((term, tf_idf))
        # sort the list based on tf-idf value in descending order
        tf_idf_list.sort(key=lambda tup: tup[1], reverse=True)
        # store list with country code as key in tf-idf dic
        tf_idf_country_dict[country_code] = tf_idf_list

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
    with open(TF_IDF_FLICKR_STORE_PATH, 'at', encoding='utf-8') as f:
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
        with open(TF_IDF_FLICKR_STORE_PATH, 'at', encoding='utf-8') as f:
            f.write(line)
    print('done.')


if __name__ == '__main__':
    flickr_w_locationref_df, country_codes = load_data()
    document_frequency, flickr_w_locationref_df = calc_document_frequency(flickr_w_locationref_df)
    if MULTIPROCESSING:
        # distribute work over cores based on country_code splits
        print(f'starting process on {cpu_count()} cores...')
        nr_country_codes_per_process = math.ceil(len(country_codes) / cpu_count())
        arguments = []
        # prepare arguments for processes
        for i in range(cpu_count()):
            # number to track process in print statements
            tracker = i + 1
            # if last process take the remaining rest
            if i == (cpu_count() - 1):
                country_codes_process_share = country_codes[(i * nr_country_codes_per_process):]
            else:
                country_codes_process_share = country_codes[(i * nr_country_codes_per_process):((i + 1) * nr_country_codes_per_process)]
            arguments.append((flickr_w_locationref_df, country_codes_process_share, tracker))
        # convert to tuple
        arguments = tuple(arguments)
        print('initiating process pooling...')
        with Pool() as pool:
            country_term_frequency_dict_list = pool.starmap(calc_term_frequency, arguments)
        # merge dictionaries that were generated by the different processes
        country_term_frequency_dict = {}
        for dict_ in country_term_frequency_dict_list:
            country_term_frequency_dict.update(dict_)
    if not MULTIPROCESSING:
        print('no multiprocessing - continuing on single core...')
        country_term_frequency_dict = calc_term_frequency(flickr_w_locationref_df, country_codes)
    tf_idf_country_dict = calc_tf_idf(flickr_w_locationref_df, document_frequency, country_term_frequency_dict)
    save_output(tf_idf_country_dict, TOP_TERMS)