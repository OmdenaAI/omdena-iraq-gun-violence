# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 22:51:14 2021

@author: shyam
"""

import praw
import json
import pandas as pd

class RedditScraper():
    
    def __init__(self, config_file, columns, search_params): 
        self.config = json.load(open(config_file, 'r'))
        self.client_id = self.config["client_id"]
        self.client_secret = self.config["client_secret"]
        self.user_agent = self.config["user_agent"]
        self.password = self.config["password"]
        self.username = self.config["username"]
        
        self.reddit = self.__create_reddit_scraper()
        
        self.columns = columns
        self.search_params = search_params
                   
        
    def __create_reddit_scraper(self):
        return praw.Reddit(client_id=self.client_id, 
                         client_secret=self.client_secret, 
                         user_agent=self.user_agent,
                         password=self.password,
                         username=self.username)
        
    
    def scrape(self, keyword, subreddit="all"):
        submissions = []
        subreddit = self.reddit.subreddit(subreddit)
        for submission in subreddit.search(keyword, **self.search_params):
            submissions.append([submission.title, submission.subreddit, submission.author, submission.url])                            
        self.final_df = pd.DataFrame(submissions,columns=self.columns[:-1])
        if self.final_df.shape[0]!=0: 
            self.final_df.loc[:, self.columns[-1]] = "Reddit" 
        else:
            pass
        
        return self.final_df

