import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_single_theme_stocks(url=None, theme=None):
    if url is None:
        url = "https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=468"
        theme = "random"

    #contentarea > div:nth-child(5) > table > tbody > tr:nth-child(1) > td.name > div > a
    #contentarea > div:nth-child(5) > table > tbody > tr:nth-child(2) > td.name > div > a
    #contentarea > div:nth-child(5) > table > tbody > tr:nth-child(16) > td.name > div > a

    response = requests.get(url)
    stocks = list()

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # tbody 요소 가져오기
        tbody = soup.select_one("#contentarea > div:nth-child(5) > table > tbody")

        if tbody:
            # tbody 안에 있는 모든 tr:nth-child() 이하의 요소 선택하기
            rows = tbody.select("tr:nth-child(n)")

            for i, row in enumerate(rows, start=1):
                # 각 행에서 a 태그 내의 텍스트 가져오기
                element = row.select_one("td.name > div > a")

                if element:
                    text = element.get_text(strip=True)
                    stocks.append(text)

                else:
                    print(f"Element {i}의 a 태그를 찾을 수 없습니다.")
        else:
            print("tbody를 찾을 수 없습니다.")
    else:
        print("웹페이지에 접근할 수 없습니다. 상태 코드:", response.status_code)

    df = pd.DataFrame({'theme': theme,'title': stocks})

    return df



if __name__ == '__main__':
    stock_list = [
        ('화이자(PFIZER)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=468'),
        ('전자결제(전자화폐)', 'https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=272')

    ]

    df = get_single_theme_stocks(theme=stock_list[0][0], url=stock_list[0][1])
    for key, val in stock_list[1:]:
        stock_df = get_single_theme_stocks(url=val, theme=key)
        df = pd.concat([df, stock_df])

    df.to_csv('themed_stocks.csv', encoding='cp949')
    print('saved csv file (themed stocks)')
