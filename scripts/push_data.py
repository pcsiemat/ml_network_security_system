"""
This script is responsible for extracting network data from CSV files and inserting it into a MongoDB database.
It includes functionality to convert CSV data to JSON format and handle database operations securely.
"""
import os    # Access environment variables and OS-level functionality
import json  # Read and write JSON data
import sys   # System-specific parameters and functions (e.g., sys.exit, sys.path)

from dotenv import load_dotenv  # Loads key=value pairs from a .env file
load_dotenv()                   # Read the .env file and add its values to the environment

MONGO_DB_URL = os.getenv("MONGO_DB_URL")  # Get the MongoDB connection string (None if not set)
#print(f"MONGO_DB_URL: {MONGO_DB_URL}")    # Print the URL to confirm it loaded (debug only)

import certifi       # Provides an up-to-date bundle of trusted CA certificates
ca = certifi.where() # Path to that CA bundle, used for secure TLS connections to MongoDB (e.g., Atlas)

import pandas as pd  # Data manipulation and analysis library
import numpy as np   # Numerical computing library
import pymongo       # MongoDB driver for Python

from ml_network_security_system.exception.exception import AppException
from ml_network_security_system.logger.log import logging

class NetworkDataExtract():
    """
    NetworkDataExtract is a class that handles the extraction of network data from CSV files and insertion into a MongoDB database.
    It provides methods to convert CSV data to JSON format and insert that data into a specified MongoDB collection."""
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise AppException(e, sys)
    def cv_to_json(self, filepath):
        """
        Convert a CSV file to JSON format.
        :param filepath: Path to the CSV file
        :return: JSON data as a list of dictionaries
        """
        try:
            data = pd.read_csv(filepath)  # Read the CSV file into a DataFrame
            data.reset_index(drop=True, inplace=True)  # Reset the index of the DataFrame
            # df:   name  age        ->  records:
            #       Ana   25             [{"name": "Ana", "age": 25},
            #       Bob   30              {"name": "Bob", "age": 30}]
            records = list(json.loads (data.T.to_json()).values())  # Convert DataFrame to JSON and load as list of dicts
            return records  # Return the JSON data
        except Exception as e:
            raise AppException(e, sys)

    def insert_data_mongodb(self, records,database, collection):
        """
        Insert data into a MongoDB collection.
        :param records: List of dictionaries to insert
        :param database: Name of the database
        :param collection: Name of the collection
        :return: None
        """
        try:
            self.database     = database  # Store the database name
            self.collection   = collection  # Store the collection name
            self.records      = records  # Store the records to insert

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)  # Connect to MongoDB using the URL and CA bundle
            self.database     = self.mongo_client[self.database]  # Access the specified database
            
            self.collection   = self.database[self.collection]  # Access the specified collection

            self.collection.delete_many({})              # vacía la colección (conserva índices)


            self.collection.insert_many(self.records)  # Insert the records into the collection
            logging.info(f"✅ Data inserted successfully into {database}.{collection}")  # Log success message
            return (len(self.records))  # Return the number of records inserted 

        except Exception as e:
            raise AppException(e, sys)  # Raise custom exception on failure



if __name__=='__main__':
    FILE_PATH = os.path.join("Network_Data", "phisingData.csv")
    #FILE_PATH = "Network_Data\phisingData.csv"
    DATABASE = "Cmatos_DB"
    Collection = "Network_Data_Collection"
    networkobj = NetworkDataExtract()
    records = networkobj.cv_to_json(filepath=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, Collection)
    print(no_of_records)