import os, sys
sys.path.append(os.getcwd()) # vscode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.ch4_etf.utils import load_close_data_of_portfolio_list
from src.ch1_clustering.utils import *
import pickle

class Portfolio():
    def __init__(self):
        self.price = load_close_data_of_portfolio_list()
        self.port_samples = 10000
        self.base_folder = os.path.join(os.getcwd(), 'portfolio')
        self.port_risks = None
        self.port_returns = None
        self.sorted_shape_idx = None
        self.sorted_risk_idx = None
        self.port_ratios = None
        self.sorted_port_df = None
        self.sorted_returns = None
        self.sorted_risks = None
        self._create_portfolio_folder()

    def _create_portfolio_folder(self):
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)

    def _calculate_portfolio_performance(self):
        '''Step 4,5. Calculate return on volatility
        Calculate returns and risks for 10000 portfolio combinations
        '''
        port_ratios = []
        port_returns = np.array([])
        port_risks = np.array([])

        for i in range(self.port_samples):
            # Portfolio ratios
            port_ratio = np.random.rand(len(self.price.columns))
            port_ratio /= port_ratio.sum()
            port_ratios.append(port_ratio)

            # Annual average returns
            total_return_rate = self.price.iloc[-1] / self.price.iloc[0]
            annual_avg_rr = total_return_rate ** (1/10)
            port_return = np.dot(port_ratio, annual_avg_rr-1)
            port_returns = np.append(port_returns, port_return)

            # Annual volatility
            annual_cov = self.price.pct_change().cov() * len(self.price) / 10
            port_risk = np.sqrt(np.dot(port_ratio.T, np.dot(annual_cov, port_ratio)))
            port_risks = np.append(port_risks, port_risk)

        # Visualization
        self._visualize_portfolio_performance(port_risks, port_returns, port_ratios)

    def _visualize_portfolio_performance(self, port_risks, port_returns, port_ratios):
        # Step 4. Visualization
        self.port_risks = port_risks
        self.port_returns = port_returns
        self.sorted_shape_idx = np.argsort(port_returns/port_risks)
        self.sorted_risk_idx = np.argsort(port_risks)
        self.port_ratios = pd.DataFrame(port_ratios)
        self.sorted_port_df = self.port_ratios.iloc[self.sorted_shape_idx[::-1]]
        self.sorted_port_df.columns = self.price.columns
        self.sorted_returns = self.port_returns[[self.sorted_port_df.index]]
        self.sorted_risks = self.port_risks[[self.sorted_port_df.index]]

    def save_optimized_portfolio_ratio(self, file='optimized_portfolio_ratio.csv'):
        '''Step 7. Save optimized portfolio ratios'''
        if not self.port_risks or not self.port_returns:
            self._calculate_portfolio_performance()

        optimized_portfolio_ratio = pd.Series(self.sorted_port_df.iloc[0], index=self.sorted_port_df.columns)
        print(f'Optimized portfolio ratios: \n{optimized_portfolio_ratio}')

        # Save to CSV
        file_path = os.path.join(self.base_folder, file)
        optimized_portfolio_ratio.to_csv(file_path)

    def _save_figure(self, filename):
        file_path = os.path.join(self.base_folder, filename)
        plt.savefig(file_path)
        plt.close()

    def run_all_visualizations(self):
        '''Run all visualization functions'''
        if not self.port_risks.any() or not self.port_returns.any():
            self._calculate_portfolio_performance()
        self.visualize_rate_compared()
        self.visualize_percent_change()
        self.visualize_return_rate()
        self.visualize_correlation_change()
        self.visualize_return_on_volatility()
        self.visualize_portfolio_ratio_by_sharp()
        self.visualize_portfolio_returns_and_volatility_by_sharp()

    def visualize_rate_compared(self):
        '''Step 1. Visualize rate compared to the base date for 4 assets'''
        price_rate = self.price.iloc[:, :4] / self.price.iloc[0, :4]

        plt.figure(figsize=(12, 4))
        sns.lineplot(data=price_rate, linewidth=0.85)
        plt.ylim((0, price_rate.max().max()))
        plt.title('Increase/decrease rate compared to the base date')
        self._save_figure('step1_rate_compared.png')

    def visualize_percent_change(self):
        '''Step 2-1. Visualize daily returns for 4 assets'''
        pcc = self.price.iloc[:, :4].pct_change().iloc[1:, :]
        for i in range(4):
            data = pcc.iloc[:, i]
            plt.subplot(int(f'22{i+1}'))
            sns.lineplot(data=data, linewidth=0.85, alpha=0.7)
            inc_rate = (data > 0).sum() / len(data) * 100
            plt.title(f'< {data.name} : Increase rate {inc_rate:.2f}% >')
            plt.axhline(y=0, color='r', linestyle='--', linewidth=0.7, alpha=0.9)

        plt.suptitle('Percent change of each asset')
        plt.tight_layout()
        self._save_figure('step2_1_percent_change.png')

    def visualize_return_rate(self):
        '''Step 2-2. Visualize final return rates for 4 assets'''
        return_rate = (self.price.iloc[-1, :4] - self.price.iloc[0, :4]) / self.price.iloc[0, :4]

        bars = sns.barplot(x=return_rate.index, y=return_rate.values, color='Blue', alpha=0.3)
        for p in bars.patches:
            bars.annotate(f'{p.get_height():.1f}%',
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 9),
                          textcoords='offset points')

        plt.title(f'Return rate of total period (%) - {(self.price.shape[0])} days')
        self._save_figure('step2_2_return_rate.png')

    def visualize_correlation_change(self):
        '''Step 3. Visualize correlation change of percent change for all assets'''
        plt.title("Correlation of all asset's percent change")
        sns.heatmap(self.price.iloc[:, :10].pct_change().corr(), cmap='Blues', linewidth=0.2, annot=True)
        self._save_figure('step3_correlation_change.png')

    def visualize_return_on_volatility(self):
        '''Step 4. Visualize return per unit risk'''
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=self.port_risks, y=self.port_returns, c=self.port_returns/self.port_risks, cmap='cool', alpha=0.85, s=20)
        sns.scatterplot(x=self.port_risks[self.sorted_shape_idx[-1:]], y=self.port_returns[self.sorted_shape_idx[-1:]], color='r', marker='^', s=500)
        sns.scatterplot(x=self.port_risks[self.sorted_risk_idx[:1]], y=self.port_returns[self.sorted_risk_idx[:1]], color='b', marker='v', s=500)
        plt.title('Return per unit risk')
        self._save_figure('step4_return_on_volatility.png')

    def visualize_portfolio_ratio_by_sharp(self):
        '''Step 5. Visualize portfolio ratio by Sharpe Ratio'''
        plt.figure(figsize=(12, 4))
        plt.stackplot(np.arange(1, len(self.sorted_port_df)+1, 1), np.array(self.sorted_port_df.T), labels=self.sorted_port_df.columns)

        plt.xlim(0, self.port_samples)
        # plt.legend(bbox_to_anchor=(1.12, 0.95))
        plt.xlabel('Ranking of Sharpe Ratio')
        plt.ylabel('Portfolio Ratio')
        plt.title('Ranking of Optimal Portfolios by Sharpe Ratio')
        self._save_figure('step5_portfolio_ratio_by_sharpe.png')

    def visualize_portfolio_returns_and_volatility_by_sharp(self):
        '''Step 6. Visualize portfolio returns and volatility by Sharpe Ratio Ranking'''
        plt.figure(figsize=(12, 4))
        plt.fill_between(x=np.arange(1, len(self.sorted_returns[0])+1, 1), y1=self.sorted_returns[0], label='return')
        plt.fill_between(x=np.arange(1, len(self.sorted_risks[0])+1, 1), y1=self.sorted_risks[0], alpha=0.3, label='risk')
        plt.xlabel('Ranking of Sharpe Ratio')
        plt.ylabel('Return & Risk')
        plt.title('Returns & Risks of Portfolio by Sharpe Ratio Ranking')
        plt.legend()
        self._save_figure('step6_portfolio_returns_and_volatility_by_sharpe.png')

    def save_to_pickle(self, file='portfolio.pkl'):
        '''Save the Portfolio instance to a pickle file'''
        file_path = os.path.join(self.base_folder, file)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f'Portfolio instance saved to {file_path}')
        
    @classmethod
    def load_from_pickle(cls, file='portfolio.pkl'):
        '''Load a Portfolio instance from a pickle file'''
        instance = cls()
        file_path = os.path.join(instance.base_folder, file)
        with open(file_path, 'rb') as f:
            loaded_portfolio = pickle.load(f)
        print(f'Portfolio instance loaded from {file_path}')
        return loaded_portfolio

if __name__ == "__main__":
    pf = Portfolio()
    pf.save_optimized_portfolio_ratio()
    pf.run_all_visualizations()
    
    # Save the instance to a pickle file
    pf.save_to_pickle()

    # Load the instance from the pickle file
    # loaded_pf = Portfolio.load_from_pickle()
    # loaded_pf = Portfolio.load_from_pickle('portfolio_1k_sample.pkl')
    # loaded_pf.visualize_portfolio_returns_and_volatility_by_sharp()
