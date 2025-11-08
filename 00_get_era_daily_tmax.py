import cdsapi

dataset = "derived-era5-single-levels-daily-statistics"

# first_year = 1950
first_year = 1992
last_year = 2024
for year in range(first_year, last_year + 1):
    print(f"working on year {year}")
    request = {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": str(year),
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "daily_statistic": "daily_maximum",
        "time_zone": "utc+00:00",
        "frequency": "6_hourly",
        # "grid": [1.0, 1.0],
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(
        target=f"D://data//ERA5//t2m_x_daily//t2m_x_daily_{year}.nc"
    )
    # client.retrieve(dataset, request).download()
