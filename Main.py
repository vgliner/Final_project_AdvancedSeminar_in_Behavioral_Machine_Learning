import Datahandler
from Google_scholar_query import *
import numpy as np
from numpy.core.numeric import NaN
from numpy import genfromtxt
import pandas as pd



def print_intro():
    intro = 'This is a practical demonstration of recommendation system for cardiologists \n'+ \
        'Final project of Advanced Seminar in Behavioral Machine Learning - 236804\n'
    print(intro)

def print_conlcusions(recommendation, recommendation_category):
    print_indx = 1
    num_of_recom_per_category = len(recommendation)/ len(recommendation_category)
    print('*********************Results***********************')
    for rec_indx, rec in enumerate(recommendation):
        print(f'*** TOPIC : {recommendation_category[rec_indx]} ***')
        for rec1 in rec:
            print(f'Recommendation # {print_indx}:\n')
            print(f'{rec1}')
            print('-------------------------------------------------------')
            print_indx +=1
    conc = '**********End of simulation************'
    print(conc)    

def perform_matrix_factorization(latent_space_dim =50, iterations= 1000):
    dh = Datahandler.DataHandler()
    data_=dh.data_matrix.to_numpy()
    data_=data_[:,1:].astype('float64') 
    print(f'{np.shape(data_)}')    
    # d= np.array([[5,NaN,1],[1,3,5],[3,5,1]])
    d2 = Datahandler.matrix_factorization(data_,latent_space_dim,iterations)
    d2.train_model(learning_rate=.1)
    d2.test_model()

def Upload_factorization_result():
    factorization_result = genfromtxt('Decomposition.csv', delimiter=',')
    return factorization_result

def Get_recommendations(factorization_result,user_id = 1, num_top_ratings =3, num_top_items = 2):
    categories = Upload_categories()
    diseases = categories['Category \ User ID']
    ratings = factorization_result[:,user_id] 
    sorted_rating_indxs= ratings.argsort()[-num_top_ratings:]
    recommendation =[]
    recommendation_category=[]
    for indx in sorted_rating_indxs:
        recommended_category= diseases[indx]
        recommendation_category.append(recommended_category)
        recommendation_per_category = Get_top_articles_based_on_item(query= recommended_category, num_of_items=num_top_items)
        recommendation.append(recommendation_per_category[:num_top_items])
    return (recommendation,recommendation_category)

def Upload_categories():
    dh = Datahandler.DataHandler()
    return dh.data_matrix


if __name__ == "__main__":
    print_intro()
    # perform_matrix_factorization(latent_space_dim =50, iterations= 100) # Procedure which takes time and results are written to file, thus if rerun- you can comment it out
    factorization_result = Upload_factorization_result()
    recommendation,recommendation_category = Get_recommendations(factorization_result,user_id = 1, num_top_ratings =3, num_top_items = 2)
    print_conlcusions(recommendation, recommendation_category)
