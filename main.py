import logging

from pprint import pprint
from fact_checker import retrieve_engine, embed_model

def main():
    response = retrieve_engine.retrieve("Climate change is a ruse")
    pprint(response)

if __name__ == '__main__': main()