# dslab

Model Contributor:
Ajitesh Srivastava (ajiteshs@usc.edu)

Author: 
Haiwen Chen

Co-author:
Kangmin Tan

We use our own epidemic model called [SI-kJalpha](https://arxiv.org/abs/2007.05180), preliminary version of which we have successfully used during [DARPA Grand Challenge 2014](https://news.usc.edu/83180/usc-engineers-earn-national-recognition-for-predicting-disease-outbreaks/). Our forecast appears on the official [CDC webpage](https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html).  Our model can consider the effect of many complexities of the epidemic process and yet be simplified to a few parameters that are learned using fast linear regressions. Therefore, our approach can learn and generate forecasts extremely quickly. On a 2 core desktop machine, our approach takes only 3.18s to tune hyper-parameters, learn parameters and generate 100 days of forecasts of reported cases and deaths for all the states in the US. The total execution time for 184 countries is 11.83s and for more than 3000 US counties is around 30s. For around 20,000 locations data for which are made available by [Google](https://github.com/GoogleCloudPlatform/covid-19-open-data), our approch takes around 10 mins.
