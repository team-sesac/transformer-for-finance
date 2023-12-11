import pandas as pd
import requests
from bs4 import BeautifulSoup

from data.create_dataset.data_generator import save_themed_stock_since_listing_date
from data.create_dataset.utils import get_listing_date


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

    df = pd.DataFrame({'Theme': theme, 'Name': stocks})

    return df


def cat_ticker_code(df):
    stock_code = pd.read_csv('stock_list.csv', index_col=0)
    merged = pd.merge(left=df, right=stock_code, how="left", on="Name")
    print('here')
    merged.to_csv('themed_stocks_with_code.csv', encoding='UTF-8', index=False)
    print('saved themed stocks with code csv file')
    return merged


def save_stock_base_csv():
    stock_list = [
        ('화이자(PFIZER)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=468'),
        ('전자결제(전자화폐)', 'https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=272'),
        ('터치패널(스마트폰/태블릿PC등)', 'https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=209'),
        ('키오스크(KIOSK)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=209'),
        ('증강현실(AR)', 'https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=289'),
        ('UAM(도심항공모빌리티)', 'https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=462'),
        ('항공기부품','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=379'),
        ('화폐/금융자동화기기(디지털화폐 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=8'),
        ('영상콘텐츠','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=127'),
        ('광고','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=59'),
        ('지능형로봇/인공지능(AI)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=99'),
        ('RFID(NFC 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=97'),
        ('가상현실(VR)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=352'),
        ('메타버스(Metaverse)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=480'),
        ('엔터테인먼트','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=128'),
        ('의료AI','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=543'),
        ('우주항공산업(누리호/인공위성 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=200'),
        ('스마트카(SMART CAR)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=332'),
        ('차량용블랙박스','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=282'),
        ('마리화나(대마)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=408'),
        ('방위산업/전쟁 및 테러','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=144'),
        ('로봇(산업용/협동로봇 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=505'),
        ('드론(Drone)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=349'),
        ('전선','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=178'),
        ('초전도체','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=537'),
        ('자율주행차','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=362'),
        ('5G(5세대 이동통신)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=373'),
        ('쿠팡(coupang)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=474'),
        ('통신장비','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=92'),
        ('영화','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=48'),
        ('스마트팩토리(스마트공장)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=392'),
        ('LCD장비','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=7'),
        ('카지노','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=47'),
        ('제4이동통신','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=292'),
        ('NI(네트워크통합)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=56'),
        ('2021 상반기 신규상장','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=473'),
        ('모듈러주택','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=524'),
        ('남-북-러 가스관사업','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=313'),
        ('코로나19(스푸트니크V)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=477'),
        ('스마트폰','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=279'),
        ('휴대폰부품','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=40'),
        ('케이블TV SO/MSO','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=70'),
        ('PCB(FPCB 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=287'),
        ('아스콘(아스팔트 콘크리트)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=402'),
        ('4차산업 수혜주','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=375'),
        ('NI(네트워크통합)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=56'),
        ('시스템반도체','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=307'),
        ('미용기기','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=513'),
        ('갤럭시 부품주','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=393'),
        ('사물인터넷','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=334'),
        ('3D 프린터','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=326'),
        ('코로나19(혈장치료/혈장치료제)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=437'),
        ('전력설비','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=123'),
        ('보안주(정보)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=55'),
        ('2023 상반기 신규상장','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=522'),
        ('MVNO(가상이동통신망사업자)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=315'),
        ('DMZ 평화공원','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=398'),
        ('카메라모듈/부품','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=41'),
        ('U-Healthcare(원격진료)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=149'),
        ('2021 하반기 신규상장','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=499'),
        ('OLED(유기 발광 다이오드)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=9'),
        ('2차전지(LFP/리튬인산철)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=503'),
        ('2차전지(장비)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=445'),
        ('창투사','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=119'),
        ('음원/음반','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=98'),
        ('보안주(물리)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=234'),
        ('페라이트','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=534'),
        ('건설기계','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=378'),
        ('건설 중소형','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=322'),
        ('유전자 치료제/분석','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=376'),
        ('클라우드 컴퓨팅','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=276'),
        ('2023 하반기 신규상장','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=535'),
        ('우크라이나 재건','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=517'),
        ('화장품','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=110'),
        ('바이오인식(생체인식)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=106'),
        ('전기차','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=227'),
        ('2022 하반기 신규상장','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=518'),
        ('강관업체(Steel pipe)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=223'),
        ('반도체 장비','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=12'),
        ('엔젤산업','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=316'),
        ('SI(시스템통합)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=17'),
        ('모바일콘텐츠(스마트폰/태블릿PC)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=270'),
        ('보톡스(보툴리눔톡신)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=397'),
        ('음성인식','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=302'),
        ('전기차(충전소/충전기)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=514'),
        ('보톡스(보툴리눔톡신)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=397'),
        ('음성인식','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=302'),
        ('바이오인식(생체인식)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=106'),
        ('마이데이터','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=507'),
        ('해운','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=36'),
        ('스마트그리드(지능형전력망)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=229'),
        ('철강 중소형','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=72'),
        ('철도','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=250'),
        ('테마파크','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=181'),
        ('음식료업종','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=113'),
        ('네옴시티','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=519'),
        ('북한 광물자원개발','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=399'),
        ('전력저장장치(ESS)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=329'),
        ('2019 하반기 신규상장','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=420'),
        ('전자파','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=18'),
        ('무선충전기술','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=321'),
        ('윤활유','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=527'),
        ('SSD','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=216'),
        ('고령화 사회(노인복지)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=311'),
        ('자동차부품','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=27'),
        ('선박평형수 처리장치','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=363'),
        ('비철금속','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=171'),
        ('재택근무/스마트워크','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=435'),
        ('2차전지','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=64'),
        ('LED장비','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=269'),
        ('국내 상장 중국기업','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=228'),
        ('니켈','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=511'),
        ('IT 대표주','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=173'),
        ('마이크로 LED','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=390'),
        ('출산장려정책','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=139'),
        ('공작기계','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=297'),
        ('리모델링/인테리어','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=13'),
        ('골프','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=496'),
        ('해저터널(지하화/지하도로 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=213'),
        ('그래핀','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=415'),
        ('삼성페이','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=348'),
        ('남북경협','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=121'),
        ('원자력발전소 해체','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=401'),
        ('종합상사','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=325'),
        ('탄소나노튜브(CNT)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=242'),
        ('맥신(MXene)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=539'),
        ('LED장비','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=269'),
        ('2020 하반기 신규상장','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=440'),
        ('바이오시밀러(복제 바이오의약품)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=241'),
        ('마이크로 LED','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=390'),
        ('플렉서블 디스플레이','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=382'),
        ('블록체인','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=387'),
        ('LED','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=33'),
        ('백신여권','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=485'),
        ('2차전지(소재/부품)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=446'),
        ('타이어','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=167'),
        ('자원개발','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=79'),
        ('수소차(연료전지/부품/충전소 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=386'),
        ('테마파크','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=181'),
        ('모바일솔루션(스마트폰)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=288'),
        ('LCD 부품/소재','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=3'),
        ('밥솥','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=341'),
        ('수자원(양적/질적 개선)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=184'),
        ('전기자전거','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=268'),
        ('셰일가스(Shale Gas)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=323'),
        ('SNS(소셜네트워크서비스)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=317'),
        ('카카오뱅크(kakao BANK)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=489'),
        ('폴더블폰','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=404'),
        ('통신','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=126'),
        ('반도체 재료/부품','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=14'),
        ('미디어(방송/신문)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=232'),
        ('원자력발전','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=205'),
        ('치매','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=377'),
        ('CCTV＆DVR','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=28'),
        ('재난/안전(지진 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=335'),
        ('태양광에너지','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=191'),
        ('제지','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=63'),
        ('폐배터리','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=500'),
        ('조선기자재','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=94'),
        ('지주사','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=111'),
        ('마켓컬리(kurly)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=481'),
        ('日제품 불매운동(수혜)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=421'),
        ('2차전지(소재/부품)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=446'),
        ('종합상사','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=325'),
        ('종합 물류','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=141'),
        ('치아 치료(임플란트 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=174'),
        ('아프리카 돼지열병(ASF)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=417'),
        ('LNG(액화천연가스)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=381'),
        ('인터넷은행','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=343'),
        ('홈쇼핑','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=38'),
        ('여행','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=147'),
        ('겨울','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=266'),
        ('핵융합에너지','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=187'),
        ('두나무(Dunamu)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=482'),
        ('재난/안전(지진 등)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=335'),
        ('온실가스(탄소배출권)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=152'),
        ('풍력에너지','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=188'),
        ('2019 상반기 신규상장','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=414'),
        ('줄기세포','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=66'),
        ('조림사업','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=400'),
        ('구제역/광우병 수혜','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=16'),
        ('정유','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=185'),
        ('스마트홈(홈네트워크)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=104'),
        ('도시가스','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=176'),
        ('화학섬유','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=319'),
        ('태블릿PC','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=280'),
        ('황사/미세먼지','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=19'),
        ('사료','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=105'),
        ('환율하락 수혜','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=31'),
        ('반도체 대표주(생산)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=155'),
        ('HBM(고대역폭메모리)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=536'),
        ('항공/저가 항공사(LCC)','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=166'),
        ('면세점','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=380'),
        ('수산','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=90'),
        ('아이폰','https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no=388')
    ]
    df = get_single_theme_stocks(theme=stock_list[0][0], url=stock_list[0][1])
    for key, val in stock_list[1:]:
        stock_df = get_single_theme_stocks(url=val, theme=key)
        df = pd.concat([df, stock_df])

    df.to_csv('themed_stocks.csv', encoding='UTF-8', index=False)
    print('saved csv file (themed stocks)')


def main_routine():
    # 기본 주식 데이터 저장
    save_stock_base_csv()
    df = pd.read_csv('themed_stocks.csv', encoding='UTF-8')
    merged_df = cat_ticker_code(df)

    # 코드 concat 데이터 저장 및 중복 제거 저장
    merged_df = pd.read_csv('themed_stocks_with_code.csv', encoding='UTF-8', dtype={'Code': object})
    df_no_duplicates = merged_df.drop_duplicates(subset='Name', keep='first')
    df_no_duplicates.to_csv('themed_stocks_with_code_no_dup.csv', encoding='UTF-8', index=False)
    merged_df = pd.read_csv('themed_stocks_with_code_no_dup.csv', encoding='UTF-8', dtype={'Code': object})

    # 주식 상장일 데이터 수집 및 저장
    get_listing_date(path='/stock_listing_date.csv')
    listing_df = pd.read_csv('stock_listing_date.csv', encoding='UTF-8', dtype={'상장일': object, '종목코드': object})

    # 테마주 데이터에 상장일 추가
    merged_df = pd.merge(left=merged_df, right=listing_df, how="left", left_on="Code", right_on="종목코드")
    merged_df.to_csv('themed_stock_all.csv', encoding='UTF-8', index=False)

    # 테마주 상장일로부터 모든 데이터 수집
    merged_df = pd.read_csv('themed_stock_all.csv', encoding='UTF-8', dtype='object')
    merged_df = merged_df.rename(columns={'상장일': 'ListingDate'})
    save_themed_stock_since_listing_date(merged_df)

    print('saved all themed stock data')


if __name__ == '__main__':
    main_routine()
