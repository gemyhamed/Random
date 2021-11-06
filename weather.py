# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests


def get_user_input():
    ''' Returns the sum of two decimal numbers in binary digits.

            Parameters:
                    None

            Returns:
                    city (str): User's input choice from a list of aviliable cities
    '''

    #always prompt the user for input if he's wrong
    while True:
        
        value = input('Please Choose which city you want to see the weather forcasts for?\n A) London\n B) Birmingham\n C) Manchister \n')
        if value in ('London', 'Birmingham','Manchister'):
            break
            
        print('\nWARNING!! Kindly choose a valid city from the provided options only!\n\n')
        
    return value


def get_data(user_input):

    ''' Returns the sum of two decimal numbers in binary digits.

                Parameters:
                        city (str): User's input choice from a list of aviliable cities to to plot the tempreature in the upcoming hours

                Returns:
                        result_dict (dictanry): A dictinory containing hourly measures in unix timestamp and the corresponding tempreature values in celesuis
        '''
    # parameters for the API Call
    api_key = '336b9d17037213d702842ea2dc6ae933'
    base_url = 'https://api.openweathermap.org/data/2.5/onecall'
    
    #encoding for each city where 1 is london, 2 is Birmingham and 3 is Manchister 
    locations = {
    'London' : (51.5230, 0.0803), 
    'Birmingham': (53.483959 ,-2.244644),
    'Manchister' : (52.489471 ,-1.898575)
    }
    
    #Choosing the city coordinates 
    city_chosen = locations[user_input]
    lat = city_chosen[0]
    long = city_chosen[1]
    
    #calling the API and catching any exception for errors in calling
    try:
        
        url_to_call = f'{base_url}?lat={lat}&lon={long}&exclude=current,daily,alerts,minutely&appid={api_key}'
        # Get request and convert to json
        response = requests.get(url_to_call).json()
    
        # Extract date and temperature from json file
        date = [response['hourly'][time]['dt'] for time in range(len(response['hourly']))]
        # extract tempreature and convert to celesuis 
        temperature = [response['hourly'][temp]['temp'] - 273.15 for temp in range(len(response['hourly']))]
        
    except:
        raise Exception('Error in API, Please check with your adminstrator for further instructions!')
    
    #create zip iterator
    zipped = zip(date, temperature)
    temp_dict = dict(zipped)
    
    return(temp_dict)



def plot_results(temp_dict,city):
    
    ''' Returns the sum of two decimal numbers in binary digits.

                Parameters:
                        temp_dict (dictanry): A dictinory containing hourly measures in unix timestamp and the corresponding tempreature values in celesuis
                        city (str): User's input choice from a list of aviliable cities

                Returns:
                        A plot of the upcoming tempreature values in the next 48 hours and their mean, min and max values.
                        
        '''


    
    # Convert dict to pandas series
    pandSeries = pd.Series(temp_dict).rename('Temperature')
    
    # Convert unix time to human readable time
    pandSeries.index = pd.to_datetime(pandSeries.index, unit='s')
    
    # Change how the time looks to be easier to understand
    pandSeries.index= pandSeries.index.strftime('%d-%b - %I %p')
    
    # Calcuate the data min,max and mean to plot it
    avg = pandSeries.mean()
    minimum = pandSeries.min()
    min_index = pandSeries.idxmin()
    maximum = pandSeries.max()
    
    #plotting
    plt.figure(figsize=(20,8))
    plt.xticks(rotation=90)
    sns.lineplot(data =pandSeries,label='Temp')
    
    #plotting the statstics lines
    plt.axhline(minimum, color='g',label='Min Temp')
    plt.axhline(avg, color='orange',label='Average Temp')
    plt.axhline(maximum, color='r',label='Max Temp ')
    
    #general formating
    plt.title('Forecasted temperature values for {} in the next 48 Hours'.format(city))
    plt.xlabel('Time and Day')
    plt.ylabel('Temperature in Celsuis')
    plt.legend()
    plt.tight_layout()
    return plt.show()

if __name__ == '__main__':
    # Ask user for input

    user_input = get_user_input()
    # Collect data
    data = get_data(user_input)
    # Display Results
    plot_results(data,user_input)
