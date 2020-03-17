# Hotel Reservation Cancellation Analysis
<br>

The project is the result of collaborative work between Jahir Miah and Mate Pocs. 
<br>

## Executive Summary
Cancellations are a major issue when it comes to accurate financial and demand forecasting in the hospitality industry, and in the lodging industry in particular. In the dataset we examined over the course of two years, about 25% of the total reservations were cancelled. Hotels try to mitigate the cost with various strategies including cancellation policies and overbooking. 
<br>

In our analysis, we put the emphasis on predicting the cancellations correctly: if we predict that a guest will not cancel, we better be sure that they actually show up. This is a conservative approach to the problem. A question that can be answered with such a model: _The estimation of Q3 proifts are X EUR based on reservations alone, but it is very volatile, can be find a conservative estimation in the form of the profits in Q3 are going to by Y EUR with at least 90% probability?_ So it is the financial accuracy question we are aiming for. If a guest was predicted to cancel but they do show up, that is not assumed to be the same magnitude of issue. Please note that the framework can be used to answer different questions as well after changing the threshold parameters.  
<br>

In our project, we build a model that predicts whether a customer who just made a reservation will cancel it before the arrival date. The underlying data is from a specific hotel in Portugal, but the methodology can be transferred. The data is complicated enough that simple exploratory data analysis cannot unveil all the connections, so we applied different Machine Learning models, the best performing one was a Random Forest Classifier. The model performs with an approximately 90% accuracy on the test data. Using the predictions can greatly improve the accuracy of forecasts the hotel makes. 
<br>

Other than accurate forecasting, the model's results can also be used to identify the key indicators of cancellations. To name the most prominent ones: 
<br>
- deposit: customers who made a non-refundable deposit are less likely to cancel
- country: customers from outside of Portugal are less likely to cancel
- lead time: the amount of time between the reservtion and the arrival date, customers who make short term (<10 days) or long-term (>90 days) are less likely to cancel

## Data Source
The analysis is based on the data published with this article by Nuno Antonio, Ana de Almeida, Luis Nunes:

https://www.sciencedirect.com/science/article/pii/S2352340918315191

The authors collected a wide scope of variables of the guests who made a reservation between 2015 and 2017 in two hotels in Portugal. Due to the nature of the data, the guest lists and the hotels are anonymised, all we know is that the hotel identified as `H1` is a resort hotel and the one identified as `H2` is a city hotel. The two data tables contain 40k and 70k observations, respectively. We randomly picked `H1` from the two and concentrated our analysis on that data. 
<br>

## Files in the Project
- `H1.csv`: original datafile of Hotel 1, downloaded from the link above
- `H1_clean.csv`: cleaned datafile of Hotel 1, used in the modelling notebook
- `H2.csv`: original datafile of Hotel 2, downloaded from the link above
- `01_dataclean.ipynb`: data cleanup and EDA
- `02_modelling.ipynb`: modelling, final model, interpretation
- `presentation.pdf`: a presentation of the project, aimed at non-technical stakeholders
