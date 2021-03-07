#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import getopt
import sys
from urllib.parse import urlparse
import time
import requests
from bs4 import BeautifulSoup

import queue

visited_urls = set()  # all urls already visited, to not visit twice
url_id = 1
url_to_id = {}
session_file = "session_output.txt"
output_file = 'output.txt'
link_file = 'link_file.txt'

url_queue = queue.Queue()  # queue

def get_out_link(base_url, content):
    all_out_links = []
    global url_id
    for a in content.find_all('a'):
        href = a.get('href')
        if not href:
            continue
        link = base_url + href
        if href.find('wiki') == -1:
            continue
        if href[-4:] in ".png .jpg .jpeg .svg":
            continue
        all_out_links.append(link)
        if link in visited_urls:
            continue
        url_queue.put(href)
        if link not in url_to_id:
            url_to_id[link] = url_id
            url_id += 1
    return all_out_links

total_counter = 1

def get_web_content(base_url, article_url):
    if article_url in visited_urls:
        return False
    visited_urls.add(article_url)
    web_url = base_url + article_url

    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
    try:
        r = requests.get(web_url, headers={'User-Agent' : USER_AGENT})
    except requests.exceptions.ConnectionError:
        print("Connection failed")
        time.sleep(1)
        return False
    if r.status_code not in (200, 404):
        print("Request failed")
        time.sleep(1)
        return False
    soup = BeautifulSoup(r.text, 'html.parser')
    content = soup.find('div', {'id':'mw-content-text'})
    if not content:
        return False
    with open(session_file, 'a') as fout:
        fout.write(web_url + " " + str(url_to_id[web_url]) + '\n')

    all_out_links = get_out_link(base_url, content)

    global total_counter
    with open(str(total_counter), 'w') as fout:
        print("write file : " + str(url_to_id[web_url]))
        for p in content.find_all('p'):
            text = p.get_text().strip()
            if text:
                fout.write(text + '\n')
            else:
                fout.write("this paragraph is empty")
    with open(link_file, 'a') as f:
        link_ids = []
        for link in all_out_links:
            link_ids.append(url_to_id[link])
        link_ids.sort()
        f.write(str(url_to_id[web_url]) + " ")
        for link_id in link_ids:
            f.write(str(link_id) + " ")
        f.write("\n")
    print(str(total_counter) + "\t" + base_url + article_url + "\t" + str(url_to_id[web_url]))
    total_counter += 1
    return True

def bfs_craw_web_pages(init_url, article_number):
    base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(init_url))
    init_url = init_url[len(base_url):]
    web_url = base_url + init_url
    global url_id
    url_to_id[web_url] = url_id
    url_id += 1
    url_queue.put(init_url)
    global total_counter
    while not url_queue.empty() and article_number >= total_counter:
        article_url = url_queue.get()
        get_web_content(base_url, article_url)


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:u:')
    except getopt.GetoptError:
        sys.exit(2)
    url = 'http://maps.latimes.com/neighborhoods/neighborhood/list/'
    article_number = 500
    for opt, arg in opts:
        if opt == '-a':
            article_number = int(arg)
        elif opt == '-u':
            url = arg
    bfs_craw_web_pages(url, article_number)
