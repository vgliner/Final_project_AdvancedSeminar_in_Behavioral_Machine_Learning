import os
from numpy.core.numeric import NaN
from numpy.lib.function_base import gradient
import pandas as pd
import numpy as np
from numpy import savetxt
import math

class DataHandler:
    
    def __init__(self, matrix_path= None):
        self.data = []
        self.matrix_path = matrix_path
        self.data_matrix = None
        if self.matrix_path == None:
            self.matrix_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'User_vs_category.csv')
        self.data_matrix = self.Upload_matrix()
        
    
    def Upload_matrix(self):
        data_matrix = pd.read_csv(self.matrix_path)    
        # print(data_matrix.head(5))
        return data_matrix

    def get_matrix_before_factorization(self):
        pass


    def update_matrix_before_factorization(self):
        pass

    

class matrix_factorization():
    def __init__(self, data, features,iterations = 1000) -> None:
        self.data = data
        self.features = features
        self.iterations = iterations
        self.user_count = data.shape[0]
        self.item_count = data.shape[1]
        self.user_features = np.random.uniform(low=0.1, high=0.9,size=(self.user_count,self.features))
        self.item_features = np.random.uniform(low=0.1, high=0.9,size=(self.features,self.item_count))

    def MSE(self):
        """
        Mean squared error function comparing dot product of user-feature row and feature-item column to user-item cell
        """
        matrix_product = np.matmul(self.user_features,self.item_features)
        self.data_ = self.data
        nan_arr= np.argwhere(np.isnan(self.data))
        if len(nan_arr):
            for nan_indx in nan_arr:
                self.data_[nan_indx[0],nan_indx[1]]=matrix_product[nan_indx[0],nan_indx[1]]
        return np.sum((self.data_-matrix_product)**2)


    def single_gradient(self, user_row, item_col, wrt_user_idx = None, wrt_item_idx = None):
        """
        Computes gradient of single user-item cell to a single user- feature or feature-item cell
        """
        if wrt_user_idx != None and wrt_item_idx != None:
            return "Too many elements"
        
        elif wrt_user_idx == None and wrt_item_idx == None:
            return "insufficient element"

        else:
            u_row = self.user_features[user_row,:]
            i_col = self.item_features[:,item_col]
            if math.isnan(self.data[user_row,item_col]):
                return 0
            ui_rating = float(self.data[user_row,item_col])
            prediction = float(np.dot(u_row,i_col))

            if wrt_user_idx != None:
                row_elem = float(i_col[wrt_user_idx])
                gradient = 2 * (ui_rating - prediction)* row_elem
            else:
                col_elem = float(u_row[wrt_item_idx])
                gradient = 2 * (ui_rating- prediction)* col_elem
            return gradient

    def user_feature_gradient(self,user_row,wrt_user_idx):
        """
        Averages the gradients of a single user-item row with respect to a single user-feature parameter
        """
        summation = 0
        for col in range(0,self.item_count):
            summation += self.single_gradient(user_row=user_row, item_col=col, wrt_user_idx=wrt_user_idx)
            # print(f'User feature: col: {col}, summation: {summation}')
        return summation/self.item_count

    def item_feature_gradient(self,item_col,wrt_item_idx):
        """
        Averages the gradients of a single user-item column with respect to a single feature-item parameter
        """
        summation = 0
        for row in range(0,self.user_count):
            summation += self.single_gradient(user_row=row, item_col=item_col, wrt_item_idx=wrt_item_idx)
            # print(f'User feature: row: {row}, summation: {summation}')
        return summation/self.user_count

    def update_user_features(self,learning_rate):
        """
        Updates every feature-item parameter according to supplied learning rate
        """
        for i in range(0, self.user_count):
            for j in range(0,self.features):
                self.user_features[i,j]+=learning_rate*self.user_feature_gradient(user_row=i, wrt_user_idx=j)

    def update_item_features(self,learning_rate):
        """
        Updates every feature-item parameter according to supplied learning rate
        """
        for i in range(0, self.features):
            for j in range(0,self.item_count):
                self.item_features[i,j]+=learning_rate*self.item_feature_gradient(item_col=j, wrt_item_idx=i)

    def train_model(self,learning_rate=0.1):
        """
        Trains model, outputting MSE cost\ loss every 50 iterations
        """
        for i in range(self.iterations):
            self.update_user_features(learning_rate=learning_rate)
            self.update_item_features(learning_rate=learning_rate)
            Err= self.MSE()
            # if i % 50 == 0:
            print(f'Iteration #{i}, MSE={Err}')
            with open("Loss.txt", "a") as myfile:
                myfile.write(f"{Err}\n")




    def test_model(self):
        print('*******************Testing***********************')
        print(f'User features: {self.user_features}')
        savetxt('User_features.csv', self.user_features, delimiter=',', fmt='%.2f')            
        print(f'Item features: {self.item_features}') 
        savetxt('Item_features.csv', self.item_features, delimiter=',', fmt='%.2f')
        print(f'Target matrix: {self.data}')
        print(f'Decomposition result: {np.dot(self.user_features,self.item_features)}')
        savetxt('Decomposition.csv', np.dot(self.user_features,self.item_features), delimiter=',', fmt='%.2f')
        print('*******************Test Done***********************')


if __name__ == "__main__":
    print('Testing data handler')
    dh = DataHandler()
    data_=dh.data_matrix.to_numpy()
    data_=data_[:,1:].astype('float64') 
    print(f'{np.shape(data_)}')    
    d= np.array([[5,NaN,1],[1,3,5],[3,5,1]])
    d2 = matrix_factorization(data_,50)
    d2.train_model(learning_rate=.1)
    d2.test_model()
