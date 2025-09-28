import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CryptoIndexFundTradingEnv(gym.Env):
    """A cryptocurrency index fund trading environment for reinforcement learning."""

    def __init__(self, data, feature_cols=None, transaction_cost=0.001, 
                 normalize_features=True, initial_cash=10000.0):
        super().__init__()

        self.transaction_cost = transaction_cost
        self.normalize_features = normalize_features
        self.initial_cash = initial_cash

        # Load and prepare data
        self._setup_data(data, feature_cols)
        self._define_spaces()

    # -------------------- Data Setup -------------------- #
    def _setup_data(self, data, feature_cols):
        self.full_data = data.copy()
        self.tickers = sorted(self.full_data['Ticker'].unique())
        self.feature_cols = feature_cols or [col for col in self.full_data.select_dtypes(include=np.number).columns if col != 'Close']

        self.data_dict = {}
        min_len = float('inf')

        for ticker in self.tickers:
            ticker_df = self.full_data[self.full_data['Ticker'] == ticker]
            min_len = min(min_len, len(ticker_df))

            features_df = ticker_df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            if self.normalize_features:
                features_df = self._normalize_features(features_df)

            self.data_dict[ticker] = {
                'features': features_df.to_numpy(dtype=np.float32),
                'close': ticker_df['Close'].to_numpy(dtype=np.float32)
            }

        self.max_steps = min_len
        self._trim_data()
        self.close_prices_matrix = self._create_price_matrix()

    def _normalize_features(self, df):
        min_vals = df.min()
        max_vals = df.max()
        return (df - min_vals) / (max_vals - min_vals + 1e-9)

    def _trim_data(self):
        for t in self.tickers:
            self.data_dict[t]['features'] = self.data_dict[t]['features'][:self.max_steps]
            self.data_dict[t]['close'] = self.data_dict[t]['close'][:self.max_steps]

    def _create_price_matrix(self):
        return np.array([self.data_dict[t]['close'] for t in self.tickers], dtype=np.float32)

    # -------------------- Spaces -------------------- #
    def _define_spaces(self):
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.tickers) + 1,), dtype=np.float32
        )

        num_features = len(self.feature_cols) * len(self.tickers)
        obs_low = np.concatenate([
            np.zeros(num_features, dtype=np.float32) if self.normalize_features else -np.inf*np.ones(num_features),
            np.zeros(len(self.tickers), dtype=np.float32),
            np.array([0.0], dtype=np.float32)
        ])
        obs_high = np.concatenate([
            np.ones(num_features, dtype=np.float32) if self.normalize_features else np.inf*np.ones(num_features),
            np.full(len(self.tickers), np.inf, dtype=np.float32),
            np.array([np.inf], dtype=np.float32)
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    # -------------------- Reset -------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = 0.0
        self.holdings = np.zeros(len(self.tickers), dtype=np.float32)
        self.portfolio_value_history = [self.initial_cash]
        self.action_history = []

        # Initialize equal-weight portfolio
        initial_weights = np.ones(len(self.tickers)) / len(self.tickers)
        initial_prices = self.close_prices_matrix[:, self.current_step]
        self.holdings = (initial_weights * self.initial_cash) / (initial_prices + 1e-9)

        # Benchmark
        self.benchmark_holdings = (self.initial_cash / len(self.tickers)) / (initial_prices + 1e-9)
        self.benchmark_value = self.initial_cash

        return self._get_state(), {}

    # -------------------- Step -------------------- #
    def step(self, action):
        action = np.array(action, dtype=np.float32).flatten()
        if len(action) != len(self.tickers) + 1:
            raise ValueError(f"Action length {len(action)} != tickers+cash {len(self.tickers)+1}")

        processed_action = self._process_action(action)
        self.action_history.append(processed_action)

        # Previous values
        prev_agent_value = self._get_portfolio_value(self.current_step)
        prev_benchmark_value = self.benchmark_value

        # Current prices
        current_prices = self.close_prices_matrix[:, self.current_step]
        self.benchmark_value = np.dot(self.benchmark_holdings, current_prices)

        # Execute trades
        current_agent_value, trade_cost = self._execute_trades(processed_action)
        self.portfolio_value_history.append(current_agent_value)

        # Reward
        reward = self._calculate_reward(current_agent_value, prev_agent_value,
                                        self.benchmark_value, prev_benchmark_value)

        # Collect state **before moving to the next step**
        state = self._get_state()

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = terminated

        return state, reward, terminated, truncated, self._get_info(current_agent_value, processed_action, trade_cost)
    # -------------------- Process Action -------------------- #
    def _process_action(self, action):
        action = np.array(action, dtype=np.float32).flatten()
        action = np.clip(action, 0.0, 1.0)
        if np.sum(action) > 0:
            action /= np.sum(action)
        else:
            action = np.ones(len(self.tickers) + 1) / (len(self.tickers) + 1)
        return action

    # -------------------- Execute Trades -------------------- #
    def _execute_trades(self, processed_action):
        crypto_prices = self.close_prices_matrix[:, self.current_step]
        total_value = self.cash + np.dot(self.holdings, crypto_prices)

        target_allocations = processed_action * total_value
        target_cash = target_allocations[0]
        target_holdings = target_allocations[1:] / (crypto_prices + 1e-9)

        # Transaction cost
        trade_amounts = np.abs(target_holdings - self.holdings)
        trade_cost = np.sum(trade_amounts * crypto_prices * self.transaction_cost)

        # Update portfolio
        self.cash = max(target_cash - trade_cost, 0.0)
        self.holdings = target_holdings

        new_portfolio_value = self.cash + np.dot(self.holdings, crypto_prices)
        return new_portfolio_value, trade_cost

    # -------------------- Reward -------------------- #
    def _calculate_reward(self, current_agent_value, prev_agent_value,
                          current_benchmark_value, prev_benchmark_value):
        agent_return = (current_agent_value - prev_agent_value) / (prev_agent_value + 1e-9)
        benchmark_return = (current_benchmark_value - prev_benchmark_value) / (prev_benchmark_value + 1e-9)
        alpha = agent_return - benchmark_return

        mode = getattr(self, "reward_mode", "absolute")
        if mode == "absolute":
            reward = agent_return
        elif mode == "mixed":
            reward = 0.7 * agent_return + 0.3 * alpha
        elif mode == "sharpe":
            turnover_penalty = 0.0
            if len(self.action_history) > 1:
                turnover_penalty = np.sum(np.abs(self.action_history[-1] - self.action_history[-2])) * 0.0005
            reward = agent_return - turnover_penalty
        else:
            raise ValueError(f"Unknown reward_mode: {mode}")

        return float(np.clip(reward, -1, 1)) * 100

    # -------------------- Info & State -------------------- #
    def _get_info(self, current_value, action, cost):
        # Use the last valid step if we've gone past max_steps
        step_idx = min(self.current_step, self.max_steps - 1)

        current_prices = self.close_prices_matrix[:, step_idx]
        initial_prices = self.close_prices_matrix[:, 0]
        initial_holdings = self.initial_cash / len(self.tickers) / (initial_prices + 1e-9)
        market_value = np.dot(initial_holdings, current_prices)
        asset_values = self.holdings * current_prices

        return {
            "portfolio_value": current_value,
            "market_value": market_value,
            "weights": action.copy(),
            "transaction_cost": cost,
            "cash": self.cash,
            "asset_values": asset_values
        }

    def _get_state(self):
        step_idx = min(self.current_step, self.max_steps - 1)
        features = np.concatenate([self.data_dict[t]['features'][self.current_step] for t in self.tickers])
        return np.concatenate([features, self.holdings, [self.cash]]).astype(np.float32)

    def _get_portfolio_value(self, step):
        prices = self.close_prices_matrix[:, step]
        return self.cash + np.dot(self.holdings, prices)