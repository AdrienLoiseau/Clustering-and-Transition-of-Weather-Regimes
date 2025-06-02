"""
API request to download ERA5 data for 500 hPa geopotential height from the year 1940.
Source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form

Followed the instructions to install the CDS API key and the cdsapi package from the link below
https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+Windows

Author: Adrien Loiseau
Creation date: 30/10/2024 09:15
Last modified: 30/10/2024
"""

import cdsapi # Used the following command to install it: pip3 install cdsapi # for Python 3


def download_dataset(data_type="geopotential_500hPa", first_year=1940, last_year=2025, months="winter", time="midnight", area="Euro_Atlantic"):
    """
    Download the ERA5 dataset for the given data type over the given area and during the given years, months and time(s) of the day.
    Source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form

    Parameters
    ----------
        - data_type (str): The type of data to download. It can be either "geopotential" or "temperature".
        - first_year (int): The first year to download the data for.
        - last_year (int): The last year to download the data for.
        - months (str): The months to download the data for. It can be either "winter" or "full_year".
        - time (str): The time of the day to download the data for. It can be either "full_day" or "midnight".
        - area (str): The area to download the data for. It can be either "Euro_Atlantic" or "North_hemisphere".

    Returns
    -------
        - None
    """
    # Define the years to download the data for
    years_array = [str(i) for i in range(first_year, last_year+1)]

    # Define the hours to download the data for
    if (time == "full_day"):
        hours_array = [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ]
    else:
        hours_array = ["00:00"]

    # Definte the area to download the data for
    if (area == "Euro_Atlantic"): 
        # Longitude: -60 to 60, Latitude: 23.5 to 66.5
        # area_array = [66.5, -60, 23.5, 60]
        area_array = [80, -80, 20, 60]
    elif (area == "North_hemisphere"):
        # Longitude: -180 to 180, Latitude: 0 to 90
        area_array = [90, -180, 0, 180]
    # elif (area == "Europe_small"):
    #     # Longitude: -25 to 50, Latitude: 25 to 80
    #     area_array = [80, -25, 25, 50]
    # elif (area == "Europe_large"):
    #     # Longitude: -25 to 50, Latitude: 25 to 90
    #     area_array = [90, -25, 25, 50]
    elif (area == "Brussels"):
        # Longitude: 4.25 to 4.5, Latitude: 50.75 to 51
        area_array = [51, 4.25, 50.75, 4.5]
    else:
        print("The given area is not defined.")
        return

    # Definte the months to download the data for
    if (months == "winter"):
        months_array = ["01", "02", "12"]
    else: # months == "full_year"
        months_array = [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ]

    if (data_type == "geopotential_500hPa"):
        dataset = "reanalysis-era5-pressure-levels" ; variable = "geopotential"
    elif (data_type == "temperature"):
        dataset = "reanalysis-era5-single-levels" ; variable = "2m_temperature"
    else:
        print("The given data type is not defined.")
        return

    request = {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": years_array,
        "month": months_array,
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": hours_array,
        "pressure_level": ["500"], # Only for geopotential but doesn't give any error for temperature
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area_array
    }

    if (first_year == last_year):
        target = f"Data/ERA5_{data_type}_{first_year}_{months}_{time}_{area}.nc"
    else:
        target = f"Data/ERA5_{data_type}_{first_year}_{last_year}_{months}_{time}_{area}.nc"

    client = cdsapi.Client()
    client.retrieve(dataset, request, target)




# download_dataset(first_year=1940, last_year=1940, time="midnight", area="North_hemisphere") # Download the daily data for the year 1940 
# download_dataset(first_year=2020, last_year=2024, time="midnight", area="North_hemisphere") # Download the daily data for the years 1940 to 2024 
# download_dataset(first_year=2022, last_year=2022, time="full_day", area="North_hemisphere") # Download the hourly data for the year 2022

# Download the daily data for the winters of the years 2020 to 2024
download_dataset(data_type="geopotential_500hPa", first_year=1940, last_year=2025, months="winter", time="midnight", area="Euro_Atlantic") # 00:25:32 - 1.01 GB
# download_dataset(data_type="geopotential_500hPa", first_year=2000, last_year=2025, months="winter", time="midnight", area="Euro_Atlantic") # 00:20:22 - 307.88 MB
# download_dataset(data_type="geopotential_500hPa", first_year=2020, last_year=2025, months="winter", time="midnight", area="Euro_Atlantic") # 00:01:28 - 63.61 MB

# download_dataset(data_type="temperature", first_year=1940, last_year=1979, months="winter", time="full_day", area="Brussels") # 00:01:28 - 63.61 MB
# download_dataset(data_type="temperature", first_year=1980, last_year=2025, months="winter", time="full_day", area="Brussels") # 00:01:28 - 63.61 MB

# # Final download (all the data) 
# download_dataset(data_type="geopotential_500hPa", first_year=1940, last_year=2024, time="midnight", area="Euro_Atlantic") # Download the daily data for the years 1940 to 2024 


"""
Product type        Reanalysis

Variable            Geopotential

Year                1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 
                    1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 
                    1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 
                    1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 
                    1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 
                    1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                    2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                    2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                    2020, 2021, 2022, 2023, 2024, 2025

Month               January, February, March, April, May, June, 
                    July, August, September, October, November, December

Day                 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
                    29, 30, 31

Time                00:00

Pressure level      500 hPa

Geographical area   North: 66.5°, West: -60°, South: 23.5°, East: 60°

Data format         NetCDF4 (Experimental)

Download format     Unarchived (not zipped if single file)

02:32:09
4.08 GB
"""