import requests
import numpy as np
from preprocessing import get_query

def main():

    s = get_query('input.csv')
    r = requests.get("http://localhost:8031/knn", params={"item": s})

    if r.status_code == 200:
        print(r.json())
    else:
        print(r.status_code)


if __name__ == "__main__":
    main()