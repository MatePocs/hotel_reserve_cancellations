# Hotel Reservation Cancellation Analysis

The project is the result of collaborative work between Jahir Miah and Mate Pocs. 

## Data Source
The analysis is based on the data published with this article:

https://www.sciencedirect.com/science/article/pii/S2352340918315191

The authors collected various variables of the guests who made a reservation between 2015 and 2017 in two hotels in Portugal. Due to the nature of the data, the guest lists and the hotels are anonymised, all we know is that the hotel identified as `H1` is a resort hotel and the one identified as `H2` is a city hotel. 

The question we want to answer: how accurately can we predict whether a guest will cancel the reservation. We concentrate on the first hotel, H1 in the database. 
<br>

Business case: roughly 25% of the reservations were cancelled, cost to the hotel, makes financial predictions unreliable. The data is complicated enough that simple exploratory data analysis cannot unveil all the connections. 
<br>

## Files in the Project
- H1.csv: original datafile of Hotel 1, downloaded from the link above
- H2.csv: original datafile of Hotel 2, downloaded from the link above
- 01_dataclean.ipynb: our work containing 
- 02_modelling.ipynb
- presentation.pdf: a presentation of the project, aimed at non-technical stakeholders
