import os, sys
sys.path.append(os.getcwd()) # vscode

from src.ch4_etf.utils import load_close_data_of_portfolio_list
from src.ch1_clustering.utils import *


class Portfolio():
    def __init__(self):
        self.price = load_close_data_of_portfolio_list()

    def _cal_return_on_volatility(self):
        '''Step 4,5. 변동성(위험) 대비 수익률 계산
        포트폴리오 비율 조합 10000개 를 찾는 계산으로 다소 시간 소요
        '''
        port_ratios = []
        port_returns = np.array([])
        port_risks = np.array([])
        for i in range(10000): # 포트폴리오 비율 조합 10000개
            # 포트폴리오 비율
            port_ratio = np.random.rand(len(self.price.columns)) # 4가지 랜덤 실수 조합
            port_ratio /= port_ratio.sum() # 합계가 1인 랜덤 실수
            port_ratios.append(port_ratio)
            
            # 연 평균 수익률
            total_return_rate = self.price.iloc[-1] / self.price.iloc[0] # 총 수익률(%)
            # total_return_rate = (price.iloc[-1] + dividends.sum()) / price.iloc[0] # 배당금 합산 총 수익률(%)
            annual_avg_rr = total_return_rate ** (1/10) # 연 (기하)평균 수익률(%)
            port_return = np.dot(port_ratio, annual_avg_rr-1) # 연 평균 포트폴리오 수익률 = 연 평균 수익률과 포트폴리오 비율의 행렬곱
            port_returns = np.append(port_returns, port_return)
            
            # 연간 리스크
            annual_cov = self.price.pct_change().cov() * len(self.price)/10 # 연간 수익률의 공분산 = 일별 수익률 공분산 * 연간 평균 거래일수
            port_risk = np.sqrt(np.dot(port_ratio.T, np.dot(annual_cov, port_ratio))) # E(Volatility) = sqrt(WT*COV*W)
            port_risks = np.append(port_risks, port_risk)
        # Step 4. 시각화
        self.port_risks = port_risks
        self.port_returns = port_returns
        self.sorted_shape_idx = np.argsort(port_returns/port_risks)
        self.sorted_risk_idx = np.argsort(port_risks)
        # Step 5. 시각화
        self.port_ratios = pd.DataFrame(port_ratios)
        self.sorted_port_df = self.port_ratios.iloc[self.sorted_shape_idx[::-1]] # 역순
        self.sorted_port_df.columns = self.price.columns
        # Step 6.
        self.sorted_returns = self.port_returns[[self.sorted_port_df.index]]
        self.sorted_risks = self.port_risks[[self.sorted_port_df.index]]

    def save_portfolio_optimized_ratio(self, file='portfolio_optimized_ratio.csv'):
        '''Step 7. 최적의 포트폴리오 비율 저장'''
        # 전처리 및 비율 찾는 함수 실행.
        if not hasattr(self, 'port_risks') or not hasattr(self, 'port_returns'):
            self._cal_return_on_volatility()

        # 최적의 포트폴리오 추출
        save_portfolio_optimized_ratio = pd.Series(self.sorted_port_df.iloc[0], index=self.sorted_port_df.columns)
        print(f'최적의 포트폴리오 비율 : \n{save_portfolio_optimized_ratio}')

        # save to csv
        file_path = os.path.join(os.getcwd(), file)
        save_portfolio_optimized_ratio.to_csv(file_path, index=False)

    def viz_rate_compared(self):
        '''Step 1. 4개 종목의 시작일로부터 증감량 비교 시각화'''
        price_rate = self.price.iloc[:,:4]/self.price.iloc[0,:4] 

        plt.figure(figsize=(12,4))
        sns.lineplot(data=price_rate, linewidth=0.85)
        plt.ylim((0, price_rate.max().max()))
        plt.title('Increase/decrease rate compared to the base date')
        plt.show()

    def viz_percent_change(self):
        '''Step 2-1. 4개 종목의 일일 수익률 시각화'''
        pcc = self.price.iloc[:,:4].pct_change().iloc[1:,:] # 첫째 날 데이터 제거(NaN)
        for i in range(4):
            data = pcc.iloc[:,i]
            plt.subplot(int(f'22{i+1}'))
            sns.lineplot(data=data, linewidth=0.85, alpha=0.7)
            inc_rate = (data > 0).sum() / len(data) * 100
            plt.title(f'< {data.name} : Increase rate {inc_rate:.2f}% >')
            plt.axhline(y=0, color='r', linestyle='--', linewidth=0.7, alpha=0.9)

        plt.suptitle('Percent change of each asset')
        plt.tight_layout()
        plt.show()

    def viz_return_rate(self):
        '''Step 2-2. 4개 종목의 최종 수익률 시각화'''
        return_rate = (self.price.iloc[-1,:4] - self.price.iloc[0,:4])/self.price.iloc[0,:4]

        bars = sns.barplot(x=return_rate.index, y=return_rate.values, color='Blue', alpha=0.3)
        for p in bars.patches:
            bars.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va='center',
                        xytext = (0,9),
                        textcoords = 'offset points')
            
        plt.title(f'Return rate of total period (%) - {(self.price.shape[0])} days')
        plt.show()

    def viz_correlation_change(self):
        '''Step 3. 일일 수익률 간 상관관계 시각화'''
        plt.title("Correlation of all asset's percent change")
        sns.heatmap(self.price.iloc[:,:10].pct_change().corr(), cmap='Blues', linewidth=0.2, annot=True)
        plt.show()

    def viz_return_on_volatility(self):
        '''Step 4. 변동성(위험) 대비 수익률 시각화'''
        plt.figure(figsize=(12,6))
        sns.scatterplot(x=self.port_risks, y=self.port_returns, c=self.port_returns/self.port_risks, cmap='cool', alpha=0.85, s=20)
        sns.scatterplot(x=self.port_risks[self.sorted_shape_idx[-1:]], y=self.port_returns[self.sorted_shape_idx[-1:]], color='r', marker='^', s=500)
        sns.scatterplot(x=self.port_risks[self.sorted_risk_idx[:1]], y=self.port_returns[self.sorted_risk_idx[:1]], color='b', marker='v', s=500)
        plt.title('Return per unit risk')
        plt.show()

    def viz_portfolio_ratio_by_sharp(self):
        '''Step 5. 샤프 지수에 따른 포트폴리오 비율 시각화'''
        plt.figure(figsize=(12,4))
        plt.stackplot(np.arange(1,len(self.sorted_port_df)+1,1), np.array(self.sorted_port_df.T), labels=self.sorted_port_df.columns)

        plt.xlim(0,10000)
        plt.legend(bbox_to_anchor=(1.12,0.95))
        plt.xlabel('Ranking of Sharpe Ratio')
        plt.ylabel('Portfolio Ratio')
        plt.title('Ranking of Optimal Portfolios by Sharpe Ratio')
        plt.show()

    def viz_portfolio_returns_and_volatility_by_sharp(self):
        '''Step 6. 샤프 지수에 따른 포트폴리오 수익률 및 변동성 시각화'''
        plt.figure(figsize=(12,4))
        plt.fill_between(x=np.arange(1,len(self.sorted_returns)+1,1), y1=self.sorted_returns.tolist(), label='return')
        plt.fill_between(x=np.arange(1,len(self.sorted_risks)+1,1), y1=self.sorted_risks.tolist(), alpha=0.3, label='risk')
        plt.xlabel('Ranking of Sharpe Ratio')
        plt.ylabel('Return & Risk')
        plt.title('Returns & Risks of Portfolio by Sharpe Ratio Ranking')
        plt.legend()
        plt.show()
    

if __name__ == "__main__":
    pf = Portfolio()
    # pf.viz_rate_compared()
    # pf.viz_percent_change()
    # pf.viz_return_rate()
    # pf.viz_correlation_change()
    # pf.viz_return_on_volatility()
    # pf.viz_portfolio_ratio_by_sharp()
    # pf.viz_portfolio_returns_and_volatility_by_sharp()
    pf.save_portfolio_optimized_ratio()
