import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import os
from datetime import datetime
import sklearn.metrics
from sklearn.metrics import mean_absolute_error
import math
from math import factorial


class MultiSILinear:
    """
    MultiSILinear Model

    Model Formulation:
        Ajitesh Srivastava
        ajiteshs@usc.edu

    Author:
        Kangmin Tan
        kangmint@usc.edu

    Modified by:
        Haiwen Chen
        haiwenc@usc.edu

    Notice that the format of data files should be comma separated txt files
    """

    def __init__(self, infection_file, pop_file, rname_file, travel_file, k, jp, alpha, data_end_date,lapath,start,region_code,forecastDays,split):
        """
        Constructor
        :param infection_file: Cumulative Infection file
        :param pop_file: Population File
        :param rname_file: Region Name File
        :param travel_file:  Travel Matrix File
        :param k: Hyperparameter
        :param jp: Hyperparameter
        :param alpha: Hyperparameter
        :param data_end_date: Ending date of the infection data, should be in the form of 'YYYY-MM-DD'
        """
        self.I = np.array(self.load_infection_data(infection_file))
        print('self.I.shape',self.I.shape)
        self.pops = np.array(self.load_population_data(pop_file))
        self.region_names = self.load_region_names(rname_file)
        self.travel_mat = np.array(self.load_travel_data(travel_file))
        self.no_travel = [np.zeros((self.travel_mat.shape))]
        self.multi_travel_mat = self.read_travels(lapath)
        self.C = calculate_C(self.multi_travel_mat,self.pops,'inandout')
        self.Cout = calculate_C(self.multi_travel_mat,self.pops,'out')
        self.Cin = calculate_C(self.multi_travel_mat,self.pops,'in')
        self.start = start
        self.zero_diagMat = removeDiagonal(self.read_travels(lapath))
        self.norm_diag = normalization(self.read_travels(lapath),self.C)
        self.norm_diagIn = normalization(self.read_travels(lapath),self.Cin)
        self.norm_diagOut = normalization(self.read_travels(lapath),self.Cout)
        self.region_code = region_code
        self.avgDiagonal = getAvgDiagonal(self.read_travels(lapath))
        self.forecastDays = forecastDays
        self.split = split


        self.T = self.I.shape[1] # Number of days
        self.region_cnt = self.I.shape[0] # Number of regions

        self.end_date = data_end_date
        # Calculate the starting date of the data
        self.start_date = pd.date_range(end=data_end_date, periods=self.T+self.forecastDays, freq='d')[0]
        # print('self.start_date',self.start_date)

        # Data Dimension Checks
        # print('len(self.pops)',len(self.pops))
        # print('self.region_cnt',self.region_cnt)
        assert(len(self.pops) == self.region_cnt)
        # print('self.travel_mat.shape[0]',self.travel_mat.shape[0])
        # print('self.travel_mat.shape[1]',self.travel_mat.shape[1])
        # print('self.region_cnt',self.region_cnt)
        assert(self.travel_mat.shape[0] == self.travel_mat.shape[1] == self.region_cnt)
        assert(len(self.region_names) == self.region_cnt)

        # Initialize Hyperparameters
        self.k = k
        self.jp = jp
        self.alpha = alpha

    def learn_betas(self, split = -1):
        """
        Learn Betas for each region
        :param split: use day [0,split) to train the model
        :return: learned betas for each region
        """
        if split == -1:
            split = self.T
        with_travel = 1
        without_travel = 0

        A1alpha = 1
        #without travel
        X0, Y0 = self.data_prep(self.I[:, 0: split], self.no_travel, self.pops, self.k, self.jp, without_travel)
        betas0,Xj0 = self.ind_beta(X0, Y0, self.alpha, self.k, split - 1, self.pops, self.jp)
        #with travel
        X, Y = self.data_prep(self.I[:, 0: split], self.multi_travel_mat, self.pops, self.k, self.jp, with_travel)
        betas,Xj = self.ind_beta(X, Y, A1alpha, self.k, split - 1, self.pops, self.jp)
        #no normalization, no diagonal
        X1, Y1 = self.data_prep(self.I[:, 0: split], self.zero_diagMat, self.pops, self.k, self.jp, with_travel)
        betas1,Xj1 = self.ind_beta(X1, Y1, A1alpha, self.k, split - 1, self.pops, self.jp)
        
        #M6 new X2 Y2
        X2, Y2 = self.data_prep(self.I[:, 0: split], self.zero_diagMat, self.pops, self.k, self.jp, with_travel)
        betas2,Xj2 = self.ind_beta(X1, Y1, 0.85, self.k, split - 1, self.pops, self.jp)
       
      
        #with normalization, no diagonal, in only
        X3, Y3 = self.data_prep(self.I[:, 0: split], self.norm_diagIn, self.pops, self.k, self.jp, with_travel)
        betas3,Xj3 = self.ind_beta(X3, Y3, self.alpha, self.k, split - 1, self.pops, self.jp)

        #with normalization, no diagonal, out only
        X4, Y4 = self.data_prep(self.I[:, 0: split], self.norm_diagOut, self.pops, self.k, self.jp, with_travel)
        betas4,Xj4 = self.ind_beta(X4, Y4, self.alpha, self.k, split - 1, self.pops, self.jp)

        #with normalization, no diagonal, inandout, beta*T(i,I)
        X5, Y5 = self.data_prep(self.I[:, 0: split], self.norm_diag, self.pops, self.k, self.jp, with_travel)
        # betas5 = self.avgDiagonal[self.region_code] * self.ind_beta(X5, Y5, self.alpha, self.k, split - 1, self.pops, self.jp) 
        betas5,Xj5 = self.ind_beta(X5, Y5, A1alpha, self.k, split - 1, self.pops, self.jp) 

        X6, Y6 = self.data_prep(self.I[:, 0: split], self.norm_diag, self.pops, self.k, self.jp, with_travel)
        # betas5 = self.avgDiagonal[self.region_code] * self.ind_beta(X5, Y5, self.alpha, self.k, split - 1, self.pops, self.jp) 
        betas6,Xj6 = self.ind_beta(X6, Y6, self.alpha, self.k, split - 1, self.pops, self.jp) 

        # print("beta 2",betas2)
        # print("beta 5", betas5)

        return betas, betas0, betas1, betas2,betas3,betas4, betas5, X2, Xj2, betas6


    def forecast(self, betas,betas0,betas1,betas2,betas3,betas4,betas5,betas6, forward, split = - 1):
        """
        make forcast based on the betas
        :param betas: learned betas
        :param forward: how many days to predict
        :param split: splitting point of train/test set
        :return: prediction for each region starting from the split point
        """
        if split == -1:
            split = self.T

        without_travel = 0
        with_travel = 1
        infections0, yt0 = self.simulate_pred(self.I[:, self.start:split], self.no_travel, betas0, self.pops, self.k, forward, self.jp, without_travel)
        infections,yt = self.simulate_pred(self.I[:, self.start:split], self.multi_travel_mat, betas, self.pops, self.k, forward, self.jp, with_travel)

        infections1, yt1 = self.simulate_pred(self.I[:, self.start:split], self.zero_diagMat, betas1, self.pops, self.k, forward, self.jp, with_travel)
        infections2, yt2 = self.simulate_pred(self.I[:, self.start:split], self.norm_diag, betas2, self.pops, self.k, forward, self.jp, with_travel)
        infections3, yt3 = self.simulate_pred(self.I[:, self.start:split], self.norm_diagIn, betas3, self.pops, self.k, forward, self.jp, with_travel)
        infections4, yt4 = self.simulate_pred(self.I[:, self.start:split], self.norm_diagOut, betas4, self.pops, self.k, forward, self.jp, with_travel)
        infections5, yt5 = self.simulate_pred(self.I[:, self.start:split], self.norm_diag, betas5, self.pops, self.k, forward, self.jp, with_travel)
        infections6, yt6 = self.simulate_pred(self.I[:, self.start:split], self.norm_diag, betas6, self.pops, self.k, forward, self.jp, with_travel)

        return infections, infections0,infections1,infections2, infections3,infections4, infections5 ,yt2, infections6


    def draw_prediction(self, predictions, predictions0, predictions1,predictions2,predictions3,predictions4,predictions5,yt2, Xj2, region_code, split):
        """
        Draw prediction result
        :param predictions: predictions for each country
        :param region_code: index of the country to show in the region array
        :param split: splitting point of train/test set
        :return: None
        """

        #stats are NOT smoothed
        getStats(self.pops, self.I,self.T, predictions2, split)
        print('pops artsort: ',np.argsort(self.pops))


        name = self.region_names[region_code]
        drange = pd.date_range(self.start_date, periods=self.T +self.forecastDays, freq='d')
        truth = self.I[region_code, :]

        #Smooth data
        truth_smooth = savitzky_golay(truth, 15, 3)
        truth = truth_smooth

        test_truth = truth[split: min(split + predictions[region_code].shape[0], self.T)]
        mape = self.compute_mape(test_truth, predictions2[region_code])
        mape2 = self.compute_mape(test_truth, predictions5[region_code])

        # smape = SMAPESymmetricMeanAbsoluteError(test_truth,predictions2[region_code])
        # rmse = RMSERootMeanSquareError(test_truth,predictions2[region_code])

        tt = np.concatenate((truth[0:split], predictions[region_code]))
        tt0 = np.concatenate((truth[0:split], predictions0[region_code]))
        tt1 = np.concatenate((truth[0:split], predictions1[region_code]))
        tt2 = np.concatenate((truth[0:split], predictions2[region_code]))
        tt3 = np.concatenate((truth[0:split], predictions3[region_code]))
        tt4 = np.concatenate((truth[0:split], predictions4[region_code]))
        tt5 = np.concatenate((truth[0:split], predictions5[region_code]))



        ttxj2 = [0]*15
        for j in Xj2[region_code]:
            ttxj2.append(np.sum(j))
        # ttxj2[30:] =  3
        # ttxj2 = ttxj2[:40]+[x * 2 for x in ttxj2][40:65] + [x * 3 for x in ttxj2][65:]
        ttxj2 = ttxj2[:40]+[x * 3 for x in ttxj2][40:65] + [x * 4 for x in ttxj2][65:]
        ttxj2 = [abs(x) for x in ttxj2]


        # print('len(X2): ',len(yt2))
        # print('len(X2[0]): ',len(yt2[0]))
        tty2 = [0]*15
        for i in yt2: #[60]
            tty2.append(i[3][region_code]+i[0][region_code]+i[1][region_code]+i[2][region_code])

        # print('tty2',tty2)
        # tty2 = np.concatenate(tty2, predictions2[region_code])
        # tty2 = np.concatenate(yt2[truth[0:split]], yt2[region_code])
        # print('truth[0:split]',truth[0:split])
        # print('len(truth[0:split])',len(truth[0:split]))

        plt.figure(figsize=(10,7))
        plt.title("Linear MultiSI Prediction on " + name.title() +' Alpha: ' + str(self.alpha) + ' Split: ' + str(split) +' Region: '+str(region_code))
        # plt.subtitle(name)

        plt.plot(drange[0:tt.shape[0]],tt, 'c', lw = 2.5, label='Prediction with Travel')
        plt.plot(drange[0:tt0.shape[0]],tt0, 'g', lw = 2.5, label='Prediction without Travel')
        plt.plot(drange[0:tt1.shape[0]],tt1, 'y', lw = 4, label='Prediction with No Normalization & Zero Diagonal ')
        plt.plot(drange[0:tt2.shape[0]],tt2, 'r', lw = 2, label='Prediction with C Normalization & Zero Diagonal ')

        plt.plot(drange[0:tt3.shape[0]],tt3, 'g', lw = 1, label='Prediction with C Normalization & Zero Diagonal (In)')
        plt.plot(drange[0:tt4.shape[0]],tt4, 'g', lw = 1, label='Prediction with C Normalization & Zero Diagonal (Out)')
        plt.plot(drange[0:tt5.shape[0]],tt5, 'b', lw = 2, label='Prediction with C Normalization & Zero Diagonal with beta*T(i,I)')

        # plt.plot(drange[0:75],tty2, 'y', lw = 3, label='Yt Plot')
        plt.plot(drange[0:self.split],ttxj2, 'g', lw = 2, label='Xj Plot')

        plt.plot(drange[0:truth.shape[0]],truth, 'k', label="Ground Truth", lw=3)

        #smoothing the data
        # truth_smooth = savitzky_golay(truth, 15, 3)
        # plt.plot(drange[0:truth_smooth.shape[0]],truth_smooth, 'b', label="Ground Truth", lw=3)
        # print('Truth Without Smooth',truth)
        # print('Truth Without Smooth len',len(truth))
        # print('Truth With Smooth',truth_smooth)
        # print('Truth With Smooth len',len(truth_smooth))
        # plt.plot(drange[split: split + predictions2[region_code].shape[0]], predictions2[region_code], 'ro', ms=4)
        plt.legend(loc=0)
        plt.ylabel('New Infections')
        # plt.xlabel('Date')
        # plt.figtext(0.2, 0.6, "MAPE C&0 =" + mape, fontsize=15)
        # plt.figtext(0.2, 0.55, "MAPE *T(i,I) =" + mape2, fontsize=15)
        # plt.figtext(0.2, 0.5, "SMAPE =" + str(smape), fontsize=15)
        # plt.figtext(0.2, 0.45, "RMSE =" + str(rmse), fontsize=15)
        plt.grid(True)

        plt.show()






    def simulate_pred(self, data_4, passengerFlow, beta_all, popu, k, horizon, jp, with_travel):
        """
        Simulate prediction based on learned betas
        All arrays parameters here are numpy arrays
        :param data_4: The cumulative infection data
        :param passengerFlow: The travel data square matrix
        :param beta_all: Learned betas for all countries
        :param popu:  Population vector
        :param k: Hyperparameter (one number)
        :param horizon: Hyperparameter (one number)
        :param jp: Hyperparameter (one number)
        :return: An array of length [horizon] represending the prediction results.
        """

        data_4_smoothed = []
        for region in data_4:
            data_4_smoothed.append(savitzky_golay(region, 15, 3))
        data_4_smoothed= np.array(data_4_smoothed)
        # print('data_4_smoothed',data_4_smoothed)
        # print('data_4_smoothed.shape',data_4_smoothed.shape)
        data_4 = data_4_smoothed

        jk = jp * k
        popu = popu.T
        end = data_4.shape[1]
        num_countries = popu.shape[0]
        infec = np.zeros((num_countries, horizon))
        Ikt1 = np.diff(data_4[:, end - jk - 1:end])
        Ikt = np.zeros((num_countries, k))
        lastinfec = data_4[:, end - 1].T
        
        ytc = np.zeros((num_countries, horizon))

        day = 0
        for t in range(horizon):
            St = np.diag(1 - lastinfec / popu)
            # print('day: ', day, 'with_travel', with_travel)
            #n1:assumption to use prev day travel data for future
            if day <self.split:
                F = passengerFlow[day] / (np.amax(passengerFlow[day]) + 1e-10)
                # print('Normal day',day)
                # print('F',F)
            else:
                F = passengerFlow[day-self.forecastDays] / (np.amax(passengerFlow[day-self.forecastDays]) + 1e-10)
                # print('Exceed day',day)
                # print('self.forecastDays',self.forecastDays)
                # print('F',F)
            if with_travel:
                day +=1
            # print('t',t,':','F:',F)
            for kk in range(k):
                Ikt[:, kk] = np.sum(Ikt1[:, int(kk * jp):int((kk + 1) * jp)], axis=1)

            Xt = np.concatenate((St @ Ikt, F @ Ikt), axis=1) #update F at each time F

            yt = np.sum(beta_all * Xt, axis=1)
            Ikt1 = Ikt1[:, 1:Ikt1.shape[1]]
            Ikt1 = np.append(Ikt1, yt[:, np.newaxis], axis=1)

            ytc[:,t] = lastinfec + yt
            lastinfec = lastinfec + yt.squeeze().T
            infec[:, t] = lastinfec

            
            # print('simulate_pred t:',t)
        # ytc = infec
        return np.absolute(infec), ytc


    def ind_beta(self, Xt, yt, alpha, k, T_tr, pop, jp):
        region_cnt = pop.shape[0]
        jk = jp * k
        beta_all = np.zeros((region_cnt, 2 * k))
        exp_mat = np.arange(T_tr - jk, 0, -1)
        alphavec = np.power(alpha, exp_mat).T
        alphamat = np.array([alphavec, ] * 2 * k)
        Xj = []
        for j in range(region_cnt):
            y = yt[0:T_tr - jk, j]
            X = np.zeros((y.shape[0], 2 * k))
            for t in range(y.shape[0]):
                thismat = Xt[t]
                X[t, :] = thismat[:, j].T
                # print('ind_beta t:',t)
            Xj.append(X)
            X_final = alphamat.T * X
            Y_final = alphavec * y
            lb = np.array([0.0, ] * 2 * k)
            ub = np.array([1.0, ] * 2 * k)

            res = lsq_linear(X_final, Y_final, bounds=(lb, ub))
            beta_all[j, :] = res.x

        
        return beta_all, Xj


    def data_prep(self, data_4, passengerFlow, pop, k, jp, with_travel):
        """
        Prepare data prediction
        All arrays parameters below are numpy arrays.
        :param data_4: Cumulative infection data.
        :param passengerFlow: Travel matrix square matrix.
        :param pop: Population vector (Should)
        :param k: Hyperparameter
        :param jp: Hyperparameter
        """
        # print('data_4',data_4)
        # print('data_4.shape',data_4.shape)
        data_4_smoothed = []
        for region in data_4:
            data_4_smoothed.append(savitzky_golay(region, 15, 3))
        data_4_smoothed= np.array(data_4_smoothed)
        # print('data_4_smoothed',data_4_smoothed)
        # print('data_4_smoothed.shape',data_4_smoothed.shape)
        data_4 = data_4_smoothed

        pop = pop.T
        maxt = data_4.shape[1]
        jk = k * jp #last kj days
        deldata = np.diff(data_4)
        pop = pop.T

        Xt = []
        yt = np.zeros((maxt - jk, pop.shape[0]))
        Ikt = np.zeros((pop.shape[0], k))

        day = 0
        #for each timestamp t, update F
        for t in range(jk, maxt - 1): 
            Ikt1 = deldata[:, t - jk:t]
            for kk in range(k):
                Ikt[:, kk] = np.sum(Ikt1[:, kk * jp:  (kk + 1) * jp], axis=1)
            St = np.diag(1.0 - (data_4[:, t] / pop))
            if day <self.split:
                # print('day: ', day)
                F = passengerFlow[day] / (np.amax(passengerFlow[day]) + 1e-10)
                # print('Normal day',day)
                # print('F',F)
            else:
                F = passengerFlow[day-self.forecastDays] / (np.amax(passengerFlow[day-self.forecastDays]) + 1e-10)
                # print('Exceed day',day)
                # print('self.forecastDays',self.forecastDays)
                # print('F',F)

            # print('t',t,':','F:',F)
            first_term = Ikt.T @ St
            second_term = Ikt.T @ F 
            # print('first_term: ',first_term)
            # print('second_term: ',second_term)
            Xt.append(np.concatenate((first_term, second_term), 0))
            yt[t - jk, :] = deldata[:, t].T
            if with_travel:
                day += 1
        # print('Xt',Xt)
        # print('yt',yt)
        return Xt, yt



    #Static methods to load data from data files
    @staticmethod
    def load_region_names(filename):
        region_names = []
        with open(filename, "r") as f:
            for line in f:
                region_names.append(line.strip())
        return region_names

    @staticmethod
    def load_population_data(filename):
        pops = []
        with open(filename, 'r') as f:
            for line in f:
                pops.append(int(line))
        return pops

    @staticmethod
    def load_infection_data(filename):
        infection_data = []
        with open(filename, 'r') as f:
            for line in f:
                timeseries = [int(x) for x in line.split(',')]
                infection_data.append(timeseries)
        return infection_data

    @staticmethod
    def load_travel_data(filename):
        travel_data = []
        with open(filename, 'r') as f:
            i = 0
            for line in f:
                one_country_travel_info = [int(x) for x in line.split(',')]
                travel_data.append(one_country_travel_info)
                i += 1
        return travel_data


    @staticmethod
    def compute_mape(actual, forcast):
        if actual.shape[0] == 0:
            return 'unknown'
        else:
            sum = 0
            for i in range(actual.shape[0]):
                sum += np.abs((actual[i] - forcast[i])/actual[i])
            return str(round(sum / actual.shape[0],4))


    @staticmethod
    def read_travels(directory):
        dates = []
        for f in os.listdir(directory):
            if validate(f[2:12]):
                dates.append(f)
        dates.sort()
        # print(dates)
        # print(len(dates))
        dateFiles = []
        for m in dates:
            dateFiles.append(load_travel_data(directory+'/'+m))
        return dateFiles
        # print('DateFiles: ')
        # print(dateFiles[0])

def getStats(pops, selfI,selfT, predictions, split):
    smapeList = []
    rmseList = []
    
    # print('pops artsort: ',np.argsort(pops))
    # print('pops',arpops)
    #this is ALL
    oneThirds = len(pops)/3
    # print("oneThirds",oneThirds)
    c = 1
    for region_code in np.argsort(pops)[::-1][:int(oneThirds)]:
         # print(pops[region_code])
         # print('region_code',region_code)
         if pops[region_code] >0:
             # if(c>twoThirds): break
             truth = selfI[region_code, :]
             test_truth = truth[split: min(split + predictions[region_code].shape[0], selfT)]
             smapeList.append(SMAPESymmetricMeanAbsoluteError(test_truth,predictions[region_code]))
             rmseList.append(RMSERootMeanSquareError(test_truth,predictions[region_code]))
             c += 1
    print('c:',c)
    print('Statistics _________________________________________________________________________')
    # print('SMAPE : ', smapeList)
    print('Avg. SMAPE : ',np.nanmean(smapeList))
    print('Median SMAPE : ',np.nanmedian(smapeList))
    # print('Max SMAPE : ',np.nanmax(smapeList))
    # print('Min SMAPE : ',np.nanmin(smapeList))
    print('Avg. RMSE : ',np.nanmean(rmseList))
    print('Median RMSE : ',np.nanmedian(rmseList))
    # print('Max RMSE : ',np.nanmax(rmseList))
    # print('Min RMSE : ',np.nanmin(rmseList))
    print('Statistics END____________________________________________________________________')

    return smapeList, rmseList

def getStatsbyDay(pops,forecastdays, selfI,selfT, predictions, split):
    maeList = []
    rmseList = []
    
    selectedRegions = []

    oneThirds = len(pops)/3
    for region_code in np.argsort(pops)[::-1][:int(oneThirds)]:
         # print(pops[region_code])
         # print('region_code',region_code)
         if pops[region_code] >0:
            selectedRegions.append(region_code)
    # forecastdays = 14
    for day in range(forecastdays):
        # print("day",day)
        # print("selfI:",selfI)
        #this has all 208 data for 1 day
        dailytruth = selfI[selectedRegions,day]
       maeList.append(MAEMeanAbsoluteError(dailytruth,predictions[selectedRegions,day]))
        rmseList.append(RMSERootMeanSquareError(dailytruth,predictions[selectedRegions,day]))

    print('Statistics _________________________________________________________________________')
    print('Avg. MAE : ',np.nanmean(maeList))
    print('Median MAE : ',np.nanmedian(maeList))

    print('Avg. RMSE : ',np.nanmean(rmseList))
    print('Median RMSE : ',np.nanmedian(rmseList))
    print('Statistics END____________________________________________________________________')

    return maeList, rmseList

def MAEMeanAbsoluteError(actual, predicted):
    return mean_absolute_error(actual, predicted)

def RMSERootMeanSquareError(actual, predicted):
    mse = sklearn.metrics.mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    return round(rmse,4)

# def smape(a, f):
#     return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + 1e-10))

def SMAPESymmetricMeanAbsoluteError(actual, predicted):
    # return str(round(100/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))),4))
    np.seterr(divide='ignore', invalid='ignore')
    print("len(actual)",len(actual))
    print("np.abs(predicted - actual)",np.abs(predicted - actual))
    print("np.abs(actual)",np.abs(actual))
    print("np.abs(predicted)",np.abs(predicted))
    print("np.sum(2 * np.abs(predicted - actual)",np.sum(2 * np.abs(predicted - actual)))
    print("1/len(actual) * np.sum(2 * np.abs(predicted - actual) ", 1/len(actual) * np.sum(2 * np.abs(predicted - actual) ))
    print("(np.abs(actual) + np.abs(predicted)))", (np.abs(actual) + np.abs(predicted)) )
    return round(1/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))),4)

def calculate_C(multi_travel_mat,pops,traveltype):
    C_all = []
    for region in range(len(pops)):
        da = 0
        Clist = []
        populationR = pops[region]
        for i in range(int(len(multi_travel_mat)/7)):
            d = 0
            weeklySum = 0
            while d<7: 
                daily = multi_travel_mat[da]
                # print('day: ',da)
                # print('col sum', daily.sum(axis=0)[region])
                # print('row sum', daily.sum(axis=1)[region])
                # print('total=', daily.sum(axis=0)[region]+daily.sum(axis=1)[region]-daily[region][region])
                if traveltype == 'inandout':
                    weeklySum += daily.sum(axis=0)[region]+daily.sum(axis=1)[region]-daily[region][region]
                elif traveltype == 'out':
                    weeklySum += daily.sum(axis=0)[region]
                elif traveltype == 'in':
                    weeklySum += daily.sum(axis=1)[region]
                d += 1
                da +=1
            # print('weeklySum=', weeklySum)
            C = populationR/weeklySum
            # print('C =', C)
            Clist.append(C)
            # if i ==0: break
        # print('Clist',Clist)
        # print('Avg Clist',np.mean(Clist))
        C_all.append(np.mean(Clist))
    # print('C_all: ',C_all)
    # print('len(C_all): ',len(C_all))
    # print('mean(C): ',traveltype,np.mean(C_all))
    return int(np.mean(C_all))

def getAvgDiagonal(multi_travel_mat):
    diag = {}
    days = len(multi_travel_mat)
    for region in range(multi_travel_mat[0].shape[0]):
        for daily in multi_travel_mat:
            diag[region] = diag.get(region,0) + daily[region][region]
    avgDiagList = [x / days for x in list(diag.values())]
    return avgDiagList


def removeDiagonal(multi_travel_mat): 
    travel_mats = multi_travel_mat
    for m in travel_mats:
        # print('old m:',m)
        np.fill_diagonal(m,0)
        # print('new m:',m)
    # print(travel_mats)
    return travel_mats

def normalization(multi_travel_mat,C):
    travel_mats = multi_travel_mat
    for m in travel_mats:
        # print('old m:',m)
        np.fill_diagonal(m,0)
        m *= C
        # print('new m:',m)
    # print(travel_mats)
    return travel_mats

def validate(date_text):
    try:
        if date_text != datetime.strptime(date_text,"%Y-%m-%d").strftime('%Y-%m-%d'):
            raise ValueError
        return True
    except ValueError:
        return False

def load_travel_data(filename):
        travel_data = []
        with open(filename, 'r') as f:
            i = 0
            for line in f:
                one_country_travel_info = [int(x) for x in line.split(',')]
                travel_data.append(one_country_travel_info)
                i += 1
        return np.array(travel_data)

def load_population_data(filename):
        pops = []
        with open(filename, 'r') as f:
            for line in f:
                pops.append(int(line))
        return pops

def load_infection_data(filename):
        infection_data = []
        with open(filename, 'r') as f:
            for line in f:
                timeseries = [int(x) for x in line.split(',')]
                infection_data.append(timeseries)
        return infection_data

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    # import numpy as np
    # from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def getForcasts(k,jp,alpha,end_date,neighborhood,start,split,forecastDays,inf_file,pop_file,region_name_file,travel_mat_file,laTravel_dir):    

 #    # Hyperparameters
 #    k = 2
 #    jp = 7
 #    alpha = 1
 #    end_date = '2020-08-30'
 #    neighborhood = 61
 #    #worked 131

 #    # start point of training set
 #    start = 0
 #    # Split point of training set and test set
 #    split = 75
 #    forecastDays = 15

    model = MultiSILinear(inf_file,pop_file,region_name_file,travel_mat_file,k,jp,alpha, end_date,laTravel_dir, start,neighborhood, forecastDays, split)
    betas, betas0, betas1, betas2,betas3,betas4,betas5, X2, Xj2,betas6 = model.learn_betas(split)
    # print('betas2',betas2)
    # print('betas shape',betas.shape)
    # print('betas5',betas5)
    # print('betas5.shape ',betas5.shape)
    # print('sorted index of model.avgDiagonal',np.argsort(model.avgDiagonal))
    # print('model.avgDiagonal.len',len(model.avgDiagonal))
    for i in range(len(model.avgDiagonal)):
        # print(model.avgDiagonal[i]/(np.amax(model.avgDiagonal)))
        #only apply on the last parameter / travel column 
        betas5[i,3] *= model.avgDiagonal[i]/(np.amax(model.avgDiagonal) )#+ 1e-10)
        betas6[i,3] *= model.avgDiagonal[i]/(np.amax(model.avgDiagonal) )#+ 1e-10)
        # betas5[i] *= model.avgDiagonal[i]/(np.amax(model.avgDiagonal)+1e-10)
    # print('New betas5',betas5)
    predictions, predictions0, predictions1, predictions2,predictions3,predictions4,predictions5,yt2,predictions6 = model.forecast(betas,betas0,betas1,betas2,betas3,betas4,betas5,betas6,forecastDays, split) # Predict n days after the training point

    return predictions, predictions0, predictions1, predictions2,predictions3,predictions4,predictions5,yt2,predictions6
    # # Draw the prediction result of region number 166
    # model.draw_prediction(predictions,predictions0, predictions1,predictions2,predictions3,predictions4,predictions5,X2,Xj2,neighborhood,split)

def getError2():
    inf_file= '../../LAdata/NewCaseInfectionData.txt'
    pop_file = '../../LAdata/population.txt'
    region_name_file = '../../LAdata/neighborhood.txt'
    travel_mat_file = '../../LATravelMatrix/MS2020-03-16.txt'
    laTravel_dir = '../../LATravelMatrix'

    k = 2
    jp = 7
    alpha = .75
    end_date = '2020-08-30'
    neighborhood = 61
    
    xsize = 5
    ysize = 105
    MAE = np.array([])
    MAE0 = np.array([])
    MAE1 = np.array([])
    MAE2 = np.array([])
    MAE3 = np.array([])
    MAE4 = np.array([])
    MAE5 = np.array([])
    week1 = [MAE,MAE0,MAE1,MAE2,MAE4,MAE5,np.array([])]
    week2 = [np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
    week3 = [np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
    week4 = [np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
    week5 = [np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]

    # RMSE, RMSE0,RMSE1,RMSE2,RMSE3,RMSE4,RMSE5 = np.zeros(35),np.zeros(35),np.zeros(35),np.zeros(35),np.zeros(35),np.zeros(35),np.zeros(35)
    getMAE = False
    days = 140
    #exclude the first Sunday
    # Sundays = [6,7,8,9,10,11]
    Sundays =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    for i in Sundays:
        # MAEsublist = np.array([])
        # MAEsublist0 = np.array([])
        # MAEsublist1 = np.array([])
        # MAEsublist2 = np.array([])
        # # MAEsublist3 = np.array([])
        # MAEsublist4 = np.array([])
        # MAEsublist5 = np.array([])

        for x in [0,1,2,3,4]:

            I = np.array(load_infection_data(inf_file))
            pop = np.array(load_population_data(pop_file))


            if(x==0 and len(week1[0])<210):
                start = i*7+x*7
                split = start+35
                forecastDays = 14
                # RMSEsublist, RMSEsublist0,RMSEsublist1,RMSEsublist2,RMSEsublist3,RMSEsublist4,RMSEsublist5 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
                predictions, predictions0, predictions1, predictions2,predictions3,predictions4,predictions5,yt2,predictions6 = getForcasts(k,jp,alpha,end_date,neighborhood,start,split,forecastDays,inf_file,pop_file,region_name_file,travel_mat_file,laTravel_dir)

                print("M1:A1 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList, rmseList = getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions, split)
                print("M2:A2 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList0, rmseList0 = getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions0, split)
                print("M3:A1+B1 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList5, rmseList5 =getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions5, split)
                print("M4:A1+B2 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList1, rmseList1 =getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions1, split)
                print("M5:AB+B1 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList4, rmseList4 =getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions4, split)
                print("M6:A2+B2 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                #marked 2 but using pred 1
                maeList2, rmseList2 =getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions2, split)
                print("M7")
                maeList6, rmseList6 = getStatsbyDay(pop,forecastDays,I,I.shape[1],predictions6,split)
                if (getMAE):
                    week1[0] =np.concatenate((week1[0], helper1(maeList[:14])), axis=None)
                    week1[1] =np.concatenate((week1[1], helper1(maeList0[:14])), axis=None)
                    week1[2] =np.concatenate((week1[2], helper1(maeList1[:14])), axis=None)
                    week1[3] =np.concatenate((week1[3], helper1(maeList2[:14])), axis=None)
                    week1[4] =np.concatenate((week1[4], helper1(maeList4[:14])), axis=None)
                    week1[5] =np.concatenate((week1[5], helper1(maeList5[:14])), axis=None)
                    week1[6] =np.concatenate((week1[6], helper5(maeList6[:14])), axis=None)
                else:
                    week1[0] =np.concatenate((week1[0], helper1(rmseList[:14])), axis=None)
                    week1[1] =np.concatenate((week1[1], helper1(rmseList0[:14])), axis=None)
                    week1[2] =np.concatenate((week1[2], helper1(rmseList1[:14])), axis=None)
                    week1[3] =np.concatenate((week1[3], helper1(rmseList2[:14])), axis=None)
                    week1[4] =np.concatenate((week1[4], helper1(rmseList4[:14])), axis=None)
                    week1[5] =np.concatenate((week1[5], helper1(rmseList5[:14])), axis=None)
                    week1[6] =np.concatenate((week1[6], helper5(rmseList6[:14])), axis=None)

            if (x==1 and len(week2[0])<210):
                start = i*7+x*7
                split = start+35
                forecastDays = 15
                # RMSEsublist, RMSEsublist0,RMSEsublist1,RMSEsublist2,RMSEsublist3,RMSEsublist4,RMSEsublist5 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
                predictions, predictions0, predictions1, predictions2,predictions3,predictions4,predictions5,yt2,predictions6 = getForcasts(k,jp,alpha,end_date,neighborhood,start,split,forecastDays,inf_file,pop_file,region_name_file,travel_mat_file,laTravel_dir)

                print("M1:A1 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList, rmseList = getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions, split)
                print("M2:A2 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList0, rmseList0 = getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions0, split)
                print("M3:A1+B1 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList5, rmseList5 =getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions5, split)
                print("M4:A1+B2 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList1, rmseList1 =getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions1, split)
                print("M5:AB+B1 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                maeList4, rmseList4 =getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions4, split)
                print("M6:A2+B2 predictions"+" Sunday: "+str(i)+" x: "+str(x))
                #marked 2 but using pred 1
                maeList2, rmseList2 =getStatsbyDay(pop,forecastDays, I, I.shape[1], predictions2, split)
                print("M7")
                maeList6, rmseList6 = getStatsbyDay(pop,forecastDays,I,I.shape[1],predictions6,split)

    # fig.gca().xaxis.set_major_formatter(mdates.DayLocator(interval=250))
    
    # plt.title("MAE vs Days")   
    # plt.plot(x,week1[0],'b', label = 'M1')
    # plt.plot(x,week1[1],'g', label = 'M2')
    # plt.plot(x,week1[2],'y', label = 'M3')
    # plt.plot(x,week1[3],'r', label = 'M4')
    # plt.plot(x,week1[4],'m', label = 'M5')
    # plt.plot(x,week1[5],'c', label = 'M6')
    fig.tight_layout(pad=1)
    if (getMAE): plt.ylabel('MAE')
    else: plt.ylabel('RMSE')
    plt.xlabel('Forecast Days')
    # plt.legend(loc=0)
    plt.show()



# Example Usage of the model
if __name__ == '__main__':

    # inf_file = '../../LAdata/infectionData.txt'
    inf_file = '../../LAdata/NewCaseInfectionData.txt'
    pop_file = '../../LAdata/population.txt'
    region_name_file = '../../LAdata/neighborhood.txt'
    travel_mat_file = '../../LATravelMatrix/MS2020-03-16.txt'
    laTravel_dir = '../../LATravelMatrix3'

    # Hyperparameters
    k = 2
    jp = 7
    alpha = 1
    end_date = '2021-03-01'
    neighborhood = 61
    #worked 131

    # start point of training set
    start = 45
    # Split point of training set and test set
    split = 75
    forecastDays = 14


    getError2()


