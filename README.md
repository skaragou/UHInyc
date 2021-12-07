## Final Report - Modeling the Urban Head Island Effect in NYC

The urban heat island effect is a phenomenon where temperatures in cities are quite higher than those found in surrounding suburban and rural areas. The phenomenon is especially acute in large dense cities during summers. The higher temperatures can be attributed to the lack of greenery such as trees, a large concentration of cement and a high-use of heat-inducing equipment, such as air conditioners. Greenery tends to lower temperatures due to the release of humidity, as well as, in the case of trees, creating a canopy for the streetscape, so as to provide shade. A concentration of cement and asphalt leads to higher average temperatures, as both materials trap heat from direct sunlight and slowly release the heat throughout the night, as surfaces slowly cool down. Lastly, the use of equipment such as air conditioners and cars, directly contribute heat to surrounding environments, which in conjunction with the other contributors to the urban heat island effect, lifts urban temperatures.

The UHI effect is important to study, as in the near future, due to climate change, summers may become warmer and cities that suffer from the effect more acutely may face upticks in heat strokes among the most vulnerable populations. The Urban Heat Island effect has been studied in many cities around the world and New York City has been one of the cities that have been studied thoroughly. This is due to the fact that there are large temperature disparities within the city itself. Specifically, neighborhoods that tend to be lower income also tend to have the highest summer temperatures. This may be due to the fact that wealthier neighborhoods usually have a larger density of trees, while poorer neighborhoods lack investment in their respective urban environment. 

### Files

* **data_processing.py:** This generates the necessary csv file to create the data for the model. The datasets necessary to run this are and the script takes around 2 hours to run:
	* ![Hyperlocal Temperature Monitoring][https://data.cityofnewyork.us/dataset/Hyperlocal-Temperature-Monitoring/qdq3-9eqn/data]
	* ![Park Property][https://data.cityofnewyork.us/City-Government/ARCHIVED-Parks-Properties/k2ya-ucmv]
	* ![Buildings][https://data.cityofnewyork.us/Housing-Development/Building-Footprints/nqwf-w8eh]
	* ![2015 Tree Census][https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/pi5s-9p35]

* **data_processing_models.py:** After the data.csv has been created, this generates smaller h5 files, which act as inputs to the models. These have been provided on GitHub.
* **final_report.py:** The final report! (Takes about 2.5-3 hours to run)
* **models.py:** Contains the models
* **plot.py:** Some graphs for the final report. Requries data.csv
* **plot_avg_model.py:** Contains functions for plots the 24h model
* **plot_max_model.py:** Contains functions for plots the max model

### Discussion and Improvements
Ultimately, this problem is very difficult with the data at hand. As seen above, the variables cannot explain the variance in temperature registered by the sensors very well. Consideration could be given to the following improvements:

* Creating a new causal diagram to include more variables that could be a factor for temperatures registered by a sensor. If it is possible to describe the variance in temperature, coefficients in the model could be more reliable causal effect estimators.

* (Yin et al. 2018) notes that land surface temperatures are inherently correlated (sensors near each other will likely have similar variation). This breaks the indepedence condition assumed on the depedent variable in linear regression. Hence, the paper uses a method called spatial lag regression, which allows for spatial autocorelation, by conditioning on surrounding temperatures when predicting a single temperature. Both models above make the independence assumption and hence not taking spatial autocorelation into account is incorrect. Thus, some of the unexplained variance could be lessened if spatial factors were added to a model. If I had more time on this project, I would experiment with spatial lag regression.

### References

* Chaohui Yin et al. “Effects of urban form on the urban heat island effect based on spatial regression model”. In: Science of The Total Environment 634 (2018), pp. 696–704. issn: 00489697. doi: https://doi.org/10.1016/j.scitotenv.2018.03.350. url: https://www.sciencedirect.com/science/article/pii/S0048969718311100.

* Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., & Rubin, D.B. (2013). Bayesian Data Analysis (3rd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16018

* Helske, Jouni. 2020. “Efficient Bayesian Generalized Linear Models with Time-Varying Coefficients: The Walker Package in r.” https://arxiv.org/abs/2009.07063.