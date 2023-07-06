import csv
import lxml
import pandas as pd

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

import random

import time
import re

USER_LOGIN = 'XXX'
USER_PASSWORD = 'XXX'

text = []
names = []
works = []
num_reactions = []

def get_and_print_profile_info(driver, profile_url):
    driver.get(profile_url)        # this will open the link

    # Extracting data from page with BeautifulSoup
    src = driver.page_source

    # Now using beautiful soup
    soup = BeautifulSoup(src, 'lxml')

    # Extracting the HTML of the complete introduction box
    # that contains the name, company name, and the location
    intro = soup.find('div', {'class': 'pv-text-details__left-panel'})

    # In case of an error, try changing the tags used here.
    name_loc = intro.find("h1")

    # Extracting the Name
    name = name_loc.get_text().strip()
    # strip() is used to remove any extra blank spaces

    works_at_loc = intro.find("div", {'class': 'text-body-medium'})

    # this gives us the HTML of the tag in which the Company Name is present
    # Extracting the Company Name
    works_at = works_at_loc.get_text().strip()

    #собираем имя мользователя и место работы
    names.append(name)
    works.append(works_at)

    POSTS_URL_SUFFIX = 'recent-activity/all/'

    time.sleep(random.uniform(0.3,1.8))

    # Get current url from browser
    cur_profile_url = driver.current_url

    # соберем текст постов и реакции для каждого пользователя
    text.append(get_and_print_user_posts(driver, cur_profile_url + POSTS_URL_SUFFIX)[0])
    num_reactions.append(get_and_print_user_posts(driver, cur_profile_url + POSTS_URL_SUFFIX)[1])

def get_and_print_user_posts(driver, posts_url):
    driver.get(posts_url)

    # Simulate scrolling to capture all posts
    SCROLL_PAUSE_TIME = random.uniform(1.2,3.0)

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    # We can adjust this number to get more posts
    NUM_SCROLLS = 5

    for i in range(NUM_SCROLLS):
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Parsing posts
    src = driver.page_source

    # Now using beautiful soup
    soup = BeautifulSoup(src, 'lxml')

    posts = soup.find_all('li', class_='profile-creator-shared-feed-update__container')

    #создадим списки для сбора постов
    text = []
    react = []

    for post_src in posts:
        post_text_div = post_src.find('div', {'class': 'feed-shared-update-v2__description-wrapper mr2'})

        if post_text_div is not None:
            post_text = post_text_div.find('span', {'dir': 'ltr'})
        else:
            post_text = None

        # If post text is found
        if post_text is not None:
            post_text = post_text.get_text().strip()
            #собираем тексты постов в список
            text.append(post_text)
            print(f'Post text: {post_text}')

        reaction_cnt = post_src.find('span', {'class': 'social-details-social-counts__reactions-count'})

        # If number of reactions is written as text
        # It has different class name
        if reaction_cnt is None:
            reaction_cnt = post_src.find('span', {'class': 'social-details-social-counts__social-proof-text'})

        if reaction_cnt is not None:
            reaction_cnt = reaction_cnt.get_text().strip()
            #собираем реакции в список
            react.append(reaction_cnt)
            print(f'Reactions: {reaction_cnt}')

    return text, react


if __name__ == '__main__':
    # start Chrome browser
    caps = DesiredCapabilities().CHROME

    caps['pageLoadStrategy'] = 'eager'

    driver = webdriver.Chrome()

    # Opening linkedIn's login page
    # NOTE: We need to turn of 2 step authentification
    driver.get("https://linkedin.com/uas/login")

    # waiting for the page to load
    time.sleep(random.uniform(3.0,6.5))

    # entering username
    username = driver.find_element(By.ID, "username")

    # Enter Your Email Address
    username.send_keys(USER_LOGIN)

    # entering password
    pword = driver.find_element(By.ID, "password")

    # Enter Your Password
    pword.send_keys(USER_PASSWORD)

    # Clicking on the log in button
    driver.find_element(By.XPATH, "//button[@type='submit']").click()

    time.sleep(100)
    
    # Open search page
    driver.get('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx')

    profile_urls = []

    NUM_PAGES_TO_PARSE = 8

    # Iterate over pages of search results
    # to collect profile urls
    for i in range(NUM_PAGES_TO_PARSE):
        search_result_links = driver.find_elements(By.CSS_SELECTOR, "div.entity-result__item a.app-aware-link")

        for link in search_result_links:
            href = link.get_attribute("href")
            if 'linkedin.com/in' in href:
                profile_urls.append(href)

        # HACK TO SEE NEXT BUTTON BY SELENIUM
        # We scroll down the page to make element visible for Selenium
        SCROLL_PAUSE_TIME = random.uniform(2.8,5.2)

        # Get scroll height
        last_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("document.body.style.zoom='10%'")
        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, (document.body.scrollHeight/2));")
            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)
            driver.execute_script("window.scrollTo(0, (document.body.scrollHeight));")
            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        driver.execute_script("document.body.style.zoom='100%'")

        next_button = driver.find_element(By.CLASS_NAME, 'artdeco-pagination__button--next')

        next_button.click()
        time.sleep(random.uniform(12.0,14.0))

    profile_urls = list(set(profile_urls))

    # Parse profile urls
    for profile_url in profile_urls:
        get_and_print_profile_info(driver, profile_url)
        time.sleep(random.uniform(120.2,220.5))

    #сохраним датасет с данными профилей
    df_profile = pd.DataFrame.from_dict({'url': profile_urls,
                                         'name': names,
                                         'job': works,
                                         'text':text,
                                         'reactions': num_reactions})


    # сохраним датасет в файл
    df_profile.to_csv(r' parsed_data.csv', index=False)

    # close the Chrome browser
    driver.quit()
