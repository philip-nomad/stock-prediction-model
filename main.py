# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import news_contents_crawler as crawler
import news_contents_sentiment as sentiment

code_lists = ['005930','066570','051910','035420']

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # start crawling
    crawler.start(code_lists)
    for code in code_lists:
        sentiment.text_processing(code)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
