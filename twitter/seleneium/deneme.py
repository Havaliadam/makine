from selenium import webdriver
from selenium.webdriver.chrome.service import Service

service=Service("C:/Users/Acer/OneDrive/Masaüstü/seleneium/chromedriver.exe")
driver=webdriver.Chrome(service=service)
driver.get("http://www.apple.com")
