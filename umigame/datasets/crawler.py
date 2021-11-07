import time
import requests
import logging
import argparse
import pandas as pd
from lxml.html import fromstring
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from bs4.element import Comment
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor, as_completed
logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class CointelegraphCrawler(object):

    SCROLL_PAUSE_TIME = 2
    BUTTON_XPATH = '/html/body/div/div/div/div[1]/main/div/div/div[3]/div[1]/div/div/div/button'
    UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'

    def __init__(self, query, driver_root_path, thread=16):
        super(CointelegraphCrawler).__init__()
        self.root_url = f'https://cointelegraph.com/tags/{query}'
        self.driver_root_path = driver_root_path
        self.service = Service(driver_root_path)
        self.thread = thread
        self.session = self.init_session()
        
    def init_session(self):
        # Create a reusable connection pool with python requests
        session = requests.Session()
        session.mount(
            'https://',
            requests.adapters.HTTPAdapter(
                pool_maxsize=self.thread,
                max_retries=3,
                pool_block=True)
        )
        return session
    
    def run(self, time_limit, save_filename=None):
        driver = webdriver.Chrome(service=self.service)
        driver.get(self.root_url)
        WebDriverWait(driver, 5)

        logger.info(f'Running selenium...')
        start = time.time()
        click_count = 0
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            end = time.time()
            eps = end - start
            if eps >= time_limit:
                break
            try:
                # Find the button and click
                driver.execute_script(
                    "arguments[0].click();", 
                    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, self.BUTTON_XPATH)))
                )
                # Scroll down to the bottom
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                time.sleep(self.SCROLL_PAUSE_TIME)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                click_count += 1
            except:
                break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        logger.info(f'Start parsing information...')
        info = self._parse(soup)
        self._news_df = pd.DataFrame(info, columns=['title', 'link', 'author', 'date'])
        self._news_df['is_news'] = self._news_df['link'].apply(lambda x: 1 if x[26:30]=='news' else 0)
        self._news_df = self._news_df.query('is_news==1').reset_index(drop=True)
        self._news_df['content'] = self._parse_content(self._news_df['link'])
        self._news_df = self._news_df.drop(['is_news'], axis=1)
        if save_filename is not None:
            self.save_to_parquet(self._news_df, filename=f'{save_filename}.parquet.gzip')

    def _parse(self, soup):
        root = soup.find('ul', attrs={'class':'posts-listing__list'})
        stories = root.find_all('li', attrs={'class':'posts-listing__item'})
        info = []
        for story in tqdm(stories, leave=False):
            header = story.find('div', class_='post-card-inline__header')
            title = header.find('span', class_='post-card-inline__title').text.strip()
            link = 'https://cointelegraph.com' + header.find('a').get('href')
            author = header.find('p', class_='post-card-inline__author').find('a').text.strip()
            date = header.find('time').get('datetime')
            info.append([title, link, author, date])
        return info

    def get_response(self, url):
        headers = {
            'user-agent': self.UA
        }
        response = self.session.get(url, headers=headers)
        eps = response.elapsed.total_seconds()
        logging.info(f"Request was completed in {eps:.4f} seconds [{response.url}]")
        if response.status_code != 200:
            logging.error("Request failed, error code %s [%s]", response.status_code, response.url)
        if 500 <= response.status_code < 600:
            # Server is overloaded, give it a break
            time.sleep(5)
        return response

    def _parse_content(self, urls):
        contents = []
        with ThreadPoolExecutor(max_workers=self.thread) as executor:
            # wrap in a list() to wait for all requests to complete
            for response in list(executor.map(self.get_response, urls)):
                try:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    block = soup.find('div', class_='post-content')
                    texts = block.find_all(text=True)
                    visible_texts = filter(self.tag_visible, texts)
                    content = " ".join(t.strip() for t in visible_texts)
                    contents.append(content)
                except:
                    contents.append('')
        return contents

    @staticmethod
    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    @property
    def news_df(self):
        return self._news_df

    def save_to_parquet(self, df, filename='cointelegraph.parquet.gzip'):
        df.to_parquet(filename, compression='gzip')

    def load_parquet(self, filename='cointelegraph.parquet.gzip'):
        df = pd.read_parquet(filename)
        return df


def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath('//tbody/tr')[:10]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
        # Grabbing IP and corresponding PORT
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.add(proxy)
    return proxies


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Crawling the Cointelegraph...')
    parser.add_argument('--query', type=str, default='bitcoin', help='keyword to search')
    parser.add_argument('--limit', type=int, default='10', help='crawl within the time limit (seconds)')
    parser.add_argument("--save", type=str2bool, nargs='?', const=True, default=False, help="Save the file")
    args = parser.parse_args()

    query = args.query
    crawler = CointelegraphCrawler(query, r"C:\Users\Yang\Desktop\Umigame\chromedriver.exe")
    crawler.run(
        time_limit=args.limit, 
        save_filename=(f'./data/cointelegraph.{query}' if args.save else None)
    )
    news_df = crawler.news_df
    news_df['length'] = news_df['content'].map(len)
    print(news_df)
    print(news_df.query('length==0'))


if __name__ == '__main__':
    main()