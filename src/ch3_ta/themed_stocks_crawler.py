import requests
from bs4 import BeautifulSoup


#contentarea_left > table.type_1.theme > tbody > tr:nth-child(4) > td.col_type1 > a
#contentarea_left > table.type_1.theme > tbody > tr:nth-child(5) > td.col_type1 > a
#contentarea_left > table.type_1.theme > tbody > tr:nth-child(36) > td.col_type1 > a
#contentarea_left > table.type_1.theme > tbody > tr:nth-child(64) > td.col_type1 > a

import requests
from bs4 import BeautifulSoup

url = "https://finance.naver.com/sise/theme.naver?&page=1"
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    # tbody 안에 있는 모든 tr 요소 선택하기
    tr_elements = soup.select("#contentarea_left > table.type_1.theme > tbody > tr")

    for i, tr_element in enumerate(tr_elements, start=1):
        # 각 tr 요소에서 a 태그 선택하기
        a_element = tr_element.select_one("td.col_type1 > a")

        if a_element:
            # 선택한 a 태그의 href 속성 가져오기
            href = a_element.get("href")

            if href:
                print(f"Element {i}의 URL: {href}")
            else:
                print(f"Element {i}의 href 속성을 찾을 수 없습니다.")
        else:
            print(f"Element {i}에서 a 태그를 찾을 수 없습니다.")
else:
    print("웹페이지에 접근할 수 없습니다. 상태 코드:", response.status_code)


def get_single_theme_stocks(url):
    if url is None:
        url = "https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=545"

    #contentarea_left > table > tbody > tr:nth-child(4) > td:nth-child(1)

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

    print(stocks)
    return stocks
