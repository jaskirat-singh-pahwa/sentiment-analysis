from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re

quote_page = 'https://www.imdb.com/search/title/?title_type=feature&year=2021-01-01,2021-12-31&sort=year,asc'

# create a webdriver object and set options for headless browsing
options = Options()
options.headless = True
driver = webdriver.Chrome(executable_path='C:/Users/ankaggarwal/Desktop/CS410/MP2.1_private/scraper_code/chromedriver'
                                          '.exe', options=options)


# uses webdriver object to execute javascript code and get dynamically loaded web content
def get_js_soup(url, driver):
    driver.get(url)
    res_html = driver.execute_script('return document.body.innerHTML')
    soup = BeautifulSoup(res_html, 'html.parser')  # beautiful soup object to be used for parsing html content
    return soup


# tidies extracted text
def process_bio(bio):
    bio = bio.encode('ascii', errors='ignore').decode('utf-8')  # removes non-ascii characters
    bio = re.sub('\s+', ' ', bio)  # replaces repeated whitespace characters with single space
    return bio


# extracts all Faculty Profile page urls from the Directory Listing Page
def scrape_imdb_page(dir_url, driver):
    print('-' * 20, 'Scraping directory page', '-' * 20)
    movie_links = []
    # execute js on webpage to load faculty listings on webpage and get ready to parse the loaded HTML
    soup = get_js_soup(dir_url, driver)
    for link_holder in soup.find_all('h3', class_='lister-item-header'):  # get list of all <div> of class 'name'
        rel_link = link_holder.find('a')['href']  # get url
        x = 'https://www.imdb.com' + rel_link.replace("?ref_", "reviews/?ref_")
        movie_links.append(x)
    return movie_links


def scrape_movie_page(fac_url, driver):
    print(fac_url)

    soup = get_js_soup(fac_url, driver)

    bio = ''
    reviews = []
    # profile_sec = soup.find('div',class_='lister-item-content')
    print('here')

    movie_name = soup.find_all('div', class_='parent')[0].find('a').get_text()
    for link_holder in soup.find_all('div', class_='lister-item-content'):
        bio = '\"' + movie_name + '\"' + ',\"' + fac_url + '\"'
        try:
            bio = bio + ',\"' + process_bio(link_holder.find('span', class_='').get_text()) + '\"'
        except:
            print('exception occurred')
        try:
            bio = bio + ',\"' + process_bio(link_holder.find('a', class_='title').get_text()) + '\"'
        except:
            print('exception occurred')
        # print(link_holder.encode("utf-8"))
        try:
            bio = bio + ',' + '\"' + process_bio(
                link_holder.find('div', class_='content').find('div').get_text()) + '\"'
        except:
            print('exception occurred')
        reviews.append(bio)
    return reviews


def write_lst(lst, file_):
    with open(file_, 'w') as f:
        for line in lst:
            for review in line:
                f.write(review)
                f.write('\n')


# k = 20
# while k < 164 :
# start = k * 10 + 1
# quote_page = quote_page + '&start='+ str(start)
# movie_links = scrape_imdb_page(quote_page,driver)
movie_links = []
movie_links.append('https://www.imdb.com/title/tt0848228/reviews/?ref_=adv_li_tt/')
movie_links.append('https://www.imdb.com/title/tt2024544/reviews/?ref_=adv_li_tt/')
movie_links.append('https://www.imdb.com/title/tt0825232/reviews/?ref_=adv_li_tt/')
movie_links.append('https://www.imdb.com/title/tt0268978/reviews/?ref_=adv_li_tt/')
movie_links.append('https://www.imdb.com/title/tt0947798/reviews/?ref_=adv_li_tt/')
final_Reviews = []
print(movie_links)
for i, link in enumerate(movie_links):
    # print ('-'*20,'Scraping Page {} movie url {}/{}'.format(k,i+1,tot_urls),'-'*20)
    reviews = scrape_movie_page(link, driver)
    final_Reviews.append(reviews)

file_name = 'reviews' + '.csv'
write_lst(final_Reviews, file_name)
