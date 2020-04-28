# -*- coding: utf-8 -*-

from selenium import webdriver
import copy
import pandas as pd

driver = webdriver.Chrome('C:\Program Files (x86)\Google\Chrome\chromedriver_win32\chromedriver.exe')

driver.get('https://forum.shiftdelete.net/threads/iphone-8-ve-8-plus-kullananlar-kulubu.522559/page-2')

comments = pd.DataFrame(columns = ['comments']) 


j=5
while(j>=1):
    if(len(comments)<20):
        index=1
        url ='https://forum.shiftdelete.net/threads/iphone-8-ve-8-plus-kullananlar-kulubu.522559/page-'+str(j)
        driver.get(url)
        ids = driver.find_elements_by_xpath("//*[contains(@id,'js-post-')]")
        comment_ids = []
        for i in ids:
            comment_ids.append(i.get_attribute('id'))
            
        for x in comment_ids:
          
            user_message = driver.find_elements_by_xpath('//*[@id="' + x +'"]/div/div[2]/div/div/div[1]/article/div[1]')[0]
            comment = user_message.text 
            comments.loc[len(comments)] = [comment]
        j=j+1
        index=index+1
    else:
        break

comments_copy = copy.deepcopy(comments)

def remove_space(s):
    return s.replace("\n"," ")

comments_copy['comments'] = comments_copy['comments'].apply(remove_space)
comments_copy.to_csv('comments.csv', header=True, sep='#')

