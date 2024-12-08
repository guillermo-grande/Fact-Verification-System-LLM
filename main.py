import logging

from pprint import pprint

from fact_checker import retrieve_engine, embed_model
from fact_checker.data_loaders import logger as db_logger

def main():
    response = retrieve_engine.retrieve("Climate change is a ruse")
    pprint(response)

if __name__ == '__main__': main()
    