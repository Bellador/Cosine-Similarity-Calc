# import dependencies
import math
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count

'''
ANACONDA ENV: geoenv
PROCESSING STEPS (do everything for flickr sunset first - testing)
1. compare countries based on cosine similarity (build term vectors) of sunset / sunrise flickr, treat phenomena seperatly
1.1 link flickr_sunset to flickr_spatialref to get country code
1.2 group flickr_sunset by country code
1.3 calculate cosine similarity for flickr_sunset between countries
1.n calculate userdays per country
'''
# define which phenomenon is analysed
MODE = 'SUNSET' #OR SUNRISE
SOURCE = 'FLICKR' # OR INSTAGRAM OR ALL

if SOURCE == 'FLICKR':
    # random sample is used to build the entire vocabulary for tf-idf. columns: userday, userday_terms, userday_season_id
    FLICKR_RANDOM1M_PATH = Path("./Semantic_analysis(2020-12-07_FlickrInstagram_random1M/flickr_random1m_terms_geotagged_grouped.csv")
    if MODE == 'SUNSET':
        # spatial reference to unique userday hashaes. columns: userday, xbin, ybin, su_a3 (country code)
        FLICKR_LOCATIONREF_PATH = Path("./Semantic_analysis/Flickr_userday_location_ref/flickr_sunset_userday_gridloc.csv")
        # actual sunset/sunrise data. columns: userday, userday_terms, userday_season_id (Multiple/Ambiguous 0, Northern spring 1, Northern summer 2, Northern fall 3, Northern winter 4, Southern spring -1, Southern summer -2, Southern fall -3, Southern winter -4
        FLICKR_PATH = Path("./Semantic_analysis/Flickr_userday_terms_raw/flickr_sunset_terms_geotagged_grouped.csv")
        # cosine similarity store path
        COSINE_SIMILARITY_STORE_PATH = Path("./Semantic_analysis/FLICKR_SUNSET_country_cosine_similarities.csv")
        # STORE PATH FOR INTERMEDIATE PRODUCT - CALCULATED TERMS PER COUNTRY FLICKR POSTS (country_term_dict)
        COUNTRY_TERM_DICT_STORE_PATH = Path("./Semantic_analysis/FLICKR_SUNSET_country_term_dict.pickle")

    elif MODE == 'SUNRISE':
        # spatial reference to unique userday hashaes. columns: userday, xbin, ybin, su_a3 (country code)
        FLICKR_LOCATIONREF_PATH = Path("./Semantic_analysis/Flickr_userday_location_ref/flickr_sunrise_userday_gridloc.csv")
        # actual sunset/sunrise data. columns: userday, userday_terms, userday_season_id (Multiple/Ambiguous 0, Northern spring 1, Northern summer 2, Northern fall 3, Northern winter 4, Southern spring -1, Southern summer -2, Southern fall -3, Southern winter -4
        FLICKR_PATH = Path("./Semantic_analysis/Flickr_userday_terms_raw/flickr_sunrise_terms_geotagged_grouped.csv")
        # cosine similarity store path
        COSINE_SIMILARITY_STORE_PATH = Path("./Semantic_analysis/FLICKR_SUNRISE_country_cosine_similarities.csv")
        # STORE PATH FOR INTERMEDIATE PRODUCT - CALCULATED TERMS PER COUNTRY FLICKR POSTS (country_term_dict)
        COUNTRY_TERM_DICT_STORE_PATH = Path("./Semantic_analysis/FLICKR_SUNRISE_country_term_dict.pickle")

elif SOURCE == 'INSTAGRAM':
    INSTAGRAM_RANDOM1M_PATH = Path("./Semantic_analysis/2020-12-07_FlickrInstagram_random1M/instagram_random1m_terms_geotagged_grouped.csv")

def calc_vocabulary():
    '''
    PROCESSING STEP 1
    '''
    # 1.1
    # load data, use 'converters={'column_name': eval}' to evaluate the columns to their designated object. Because dataframe
    # was saved as CSV, therefore text, a stored list or series must be converted back otherwise it will appear as string
    flickr_sunset_df = pd.read_csv(FLICKR_PATH)
    flickr_locationref_sunset_df = pd.read_csv(FLICKR_LOCATIONREF_PATH)
    # merge dataframes basde on userday hash
    flickr_sunset_w_locationref_df = flickr_sunset_df.merge(flickr_locationref_sunset_df, how='left', on='userday')
    # 1.2 retrieve unique country codes for iteration
    country_codes = flickr_sunset_w_locationref_df['su_a3'].unique()
    # remove nan country codes from array (issues with further processing) - index value 17
    country_codes = np.delete(country_codes, 17)
    # create dataframe which holds cosine similarities between countries
    countries_cosine_similarity_df = pd.DataFrame(index=country_codes, columns=country_codes)
    # create dict that holds the terms for all posts inside one country based on which cosine similarity will be calcualted
    country_term_dict = {}
    print(f'column dtypes: {flickr_sunset_w_locationref_df.dtypes}')
    print('preprosses userday terms...')
    flickr_sunset_w_locationref_df['userday_terms'] = flickr_sunset_w_locationref_df['userday_terms'].apply(lambda x: x.strip('{}'))
    print('build entire corpus vocabulary (set)...')
    corpus_vocabulary_str = ','.join(flickr_sunset_w_locationref_df['userday_terms'])
    # convert to set for unique values and then back to access the index function later on
    corpus_vocabulary_set = list(set(corpus_vocabulary_str.split(',')))
    corpus_vocabulary_set_len = len(corpus_vocabulary_set)
    print(f'len corpus vocabulary (set): {corpus_vocabulary_set_len}')
    print('build vocabulary for all countries...')
    for country_index, country_code in enumerate(country_codes):
        country_df = flickr_sunset_w_locationref_df[flickr_sunset_w_locationref_df['su_a3'] == country_code]
        # drop duplicates in unique userdays (necessary? should already be unique)
        # # 1.n calculate unique userdays per country and display
        print(f'Unique userdays: {country_code}    :      {len(country_df.index.values)}')
        # build country vocabulary
        country_vocabulary_str = ','.join(country_df['userday_terms'])
        country_vocabulary_list = country_vocabulary_str.split(',')
        # create Counter object for country terms, therefore the count number of each term is acquired
        country_vocabulary_counter = Counter(country_vocabulary_list)
        # store in dictionary
        country_term_dict[country_code] = country_vocabulary_counter
    return country_term_dict, corpus_vocabulary_set, corpus_vocabulary_set_len, country_codes, countries_cosine_similarity_df


def calc_term_vector(country_term_dict, corpus_vocabulary_set, corpus_vocabulary_set_len, country_codes):
    # create dict that holds the terms_VECTORES for all post
    country_vector_dict = {}
    # create the vector of all posts based on the overall corpus vocabulary
    print('calculating term vectors...')
    for index, country_code in enumerate(country_codes, 1):
        print(f'progress: {index} of {len(country_codes)}\n')
        country_terms_counter = country_term_dict[country_code]
        # create a blueprint country_vector with the length of the entire corpus vocabulary and default value of 0
        country_vector = [0] * corpus_vocabulary_set_len
        # iterate over country term counter object
        country_terms_len = len(country_terms_counter.keys())
        for index2, (term, frequency) in enumerate(country_terms_counter.items()):
            print(f'\r{index2} of {country_terms_len}', end='')
            # find index of term in entire vocabulary coprus
            vocabulary_termindex = corpus_vocabulary_set.index(term)
            # replace 0 at given index with frequency of Counter object for given term
            country_vector[vocabulary_termindex] = frequency
        # convert to numpy array
        country_vector = np.array(country_vector)
        # reshape the vector to fit the sklearn cosine similarity function
        country_vector = country_vector.reshape(1, -1)
        country_vector_dict[country_code] = country_vector
    return country_vector_dict

def calc_cosine_similarity(country_vector_dict, countries_cosine_similarity_df):
    # 1.3 calculate cosine similarity between countries by iterating over the country_term_dict and assigning it to the countries_cosine_similarity dataframe
    print('calculating cosine similarity between country term vectors')
    for index, country_code_1 in enumerate(country_codes, 1):
        print(f'progress: {index} of {len(country_codes)}')
        country_vector_1 = country_vector_dict[country_code_1]
        for country_code_2 in country_codes:
            if country_code_1 != country_code_2:
                country_vector_2 = country_vector_dict[country_code_2]
                cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(country_vector_1, Y=country_vector_2, dense_output=True)
                # extract cosine similarity out of the lists in which it is contained
                try:
                    cosine_similarity = cosine_similarity[0][0]
                except Exception as e:
                    print(f'Cosine Similarity extraction fail: {e}')
            else:
                cosine_similarity = 1
            # add cosine similarity to dataframe
            countries_cosine_similarity_df.loc[country_code_1, country_code_2] = cosine_similarity
    # save cosine_similarity_df
    print(f'saving cosine similarities under: {COSINE_SIMILARITY_STORE_PATH}')
    countries_cosine_similarity_df.to_csv(COSINE_SIMILARITY_STORE_PATH)
    return countries_cosine_similarity_df


if __name__ == '__main__':
    print(f'starting process on {cpu_count()} cores...')
    # cosine_similarity dict is still empty here, was only initialised
    country_term_dict, corpus_vocabulary_set, corpus_vocabulary_set_len, country_codes, countries_cosine_similarity_df = calc_vocabulary()
    # dump country_term_dict as pickl format
    try:
        with open(COUNTRY_TERM_DICT_STORE_PATH, 'wb') as handle:
            pickle.dump(country_term_dict, handle)
    except Exception as e:
        print(f'Error: {e}')
        print('Could not pickle country_term_dict...')
        print('Continuing...')
    # distribute work over cores based on country_code splits
    nr_country_codes_per_process = math.ceil(len(country_codes) / cpu_count())
    arguments = []
    # prepare arguments for processes
    for i in range(cpu_count()):
        # if last process take the remaining rest
        if i == (cpu_count()-1):
            country_codes_process_share = country_codes[(i * nr_country_codes_per_process):]
        else:
            country_codes_process_share = country_codes[(i*nr_country_codes_per_process):((i+1)*nr_country_codes_per_process)]
        arguments.append((country_term_dict, corpus_vocabulary_set, corpus_vocabulary_set_len, country_codes_process_share))
    # convert to tuple
    arguments = tuple(arguments)
    with Pool() as pool:
        country_vector_dict_list = pool.starmap(calc_term_vector, arguments)
    # merge dictionaries that were generated by the different processes
    country_vector_dict = {}
    for dict_ in country_vector_dict_list:
        country_vector_dict.update(dict_)
    # compute the cosine similarity between all country specific term vectors
    countries_cosine_similarity_df = calc_cosine_similarity(country_vector_dict, countries_cosine_similarity_df)