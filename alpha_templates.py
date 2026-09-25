"""
Alpha Templates: Purpose-built tree generators targeting specific market scenarios.
Inspired by WorldQuant Brain Miner's factory architecture and quantitative finance archetypes.
"""
import random
from typing import List, Optional, Dict, Any
from ast_node import ASTNode

# Helper node builders
def f(name: str) -> ASTNode:
    return ASTNode(type='field', value=name)

def c(val: Any) -> ASTNode:
    return ASTNode(type='constant', value=str(val))

def op(name: str, *children: ASTNode) -> ASTNode:
    return ASTNode(type='operator', value=name, children=list(children))


class AlphaTemplateGenerator:
    """
    Generates alpha formula AST trees based on established quantitative trading templates:
    1. Mean Reversion (overbought / oversold, regression to mean)
    2. Momentum / Trend Following (dual horizon crossover, velocity)
    3. Volatility Breakout & Contraction (vol ratios, range expansion)
    4. Volume-Price Divergence (flow confirmation, capitulation)
    5. Dip & Rebound (deep oversold with volume or short-term reversal confirmation)
    6. Factor Combos (combining complementary archetypes)
    """

    SCENARIOS = [
        "mean_reversion",
        "momentum",
        "volatility_breakout",
        "volume_price",
        "dip_rebound",
        "factor_combo"
    ]

    SHORT_WINDOWS = [3, 5, 8, 10, 15]
    MEDIUM_WINDOWS = [20, 30, 40, 60]
    LONG_WINDOWS = [90, 120, 180, 240]

    def __init__(self, valid_fields: Optional[List[str]] = None):
        self.valid_fields = valid_fields or ["open", "high", "low", "close", "volume", "vwap"]
        self.price_fields = [fld for fld in self.valid_fields if fld in ["open", "high", "low", "close", "vwap"]] or ["close"]

    def get_available_scenarios(self) -> List[str]:
        return list(self.SCENARIOS)

    def generate(self, scenario: Optional[str] = None, apply_wrappers: bool = True) -> ASTNode:
        """
        Generates an alpha AST tree for the specified scenario (or random scenario).
        Wraps in cross-sectional ranking and smoothing if apply_wrappers is True.
        """
        if scenario is None or scenario not in self.SCENARIOS:
            weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
            scenario = random.choices(self.SCENARIOS, weights=weights, k=1)[0]

        if scenario == "mean_reversion":
            core = self._build_mean_reversion()
        elif scenario == "momentum":
            core = self._build_momentum()
        elif scenario == "volatility_breakout":
            core = self._build_volatility_breakout()
        elif scenario == "volume_price":
            core = self._build_volume_price()
        elif scenario == "dip_rebound":
            core = self._build_dip_rebound()
        elif scenario == "factor_combo":
            core = self._build_factor_combo()
        else:
            core = self._build_mean_reversion()

        if apply_wrappers:
            core = self._apply_standard_wrappers(core)

        return core

    # --- 1. Mean Reversion Archetype ---
    def _build_mean_reversion(self) -> ASTNode:
        field = random.choice(self.price_fields)
        window = random.choice(self.SHORT_WINDOWS + self.MEDIUM_WINDOWS)
        variant = random.randint(1, 5)

        if variant == 1:
            # Reversion of standardized z-score
            return op('reverse', op('ts_zscore', f(field), c(window)))
        elif variant == 2:
            # Reversion of stochastic range position (ts_scale maps to [0, 1])
            return op('reverse', op('ts_scale', f(field), c(window)))
        elif variant == 3:
            # Ratio to rolling average (mean-reversion of price relative to its moving average)
            return op('reverse', op('divide', f(field), op('ts_mean', f(field), c(window))))
        elif variant == 4:
            # Distance from average (av_diff) inverted
            return op('reverse', op('ts_av_diff', f(field), c(window)))
        else:
            # Bollinger-style dip: (mean - price) / std_dev
            mean_node = op('ts_mean', f(field), c(window))
            std_node = op('ts_std_dev', f(field), c(window))
            diff_node = op('subtract', mean_node, f(field))
            return op('divide', diff_node, std_node)

    # --- 2. Momentum Archetype ---
    def _build_momentum(self) -> ASTNode:
        field = random.choice(self.price_fields)
        short_w = random.choice(self.SHORT_WINDOWS)
        long_w = random.choice([w for w in self.MEDIUM_WINDOWS if w > short_w] or [20])
        variant = random.randint(1, 5)

        if variant == 1:
            # Dual Moving Average Crossover (Fast MA - Slow MA)
            fast = op('ts_mean', f(field), c(short_w))
            slow = op('ts_mean', f(field), c(long_w))
            return op('subtract', fast, slow)
        elif variant == 2:
            # Dual Decay Linear (LWMA crossover - reacts faster than simple MA)
            fast = op('ts_decay_linear', f(field), c(short_w))
            slow = op('ts_decay_linear', f(field), c(long_w))
            return op('subtract', fast, slow)
        elif variant == 3:
            # Time-series delta (momentum of return / change over horizon)
            return op('ts_delta', f(field), c(short_w))
        elif variant == 4:
            # Relative rate of change: delta(field, w) / delay(field, w)
            delta = op('ts_delta', f(field), c(short_w))
            prev = op('ts_delay', f(field), c(short_w))
            return op('divide', delta, prev)
        else:
            # Volatility-adjusted momentum (Sharpe ratio of time-series return)
            delta = op('ts_delta', f(field), c(short_w))
            vol = op('ts_std_dev', f(field), c(long_w))
            return op('divide', delta, vol)

    # --- 3. Volatility Breakout Archetype ---
    def _build_volatility_breakout(self) -> ASTNode:
        field = random.choice(self.price_fields)
        short_w = random.choice(self.SHORT_WINDOWS)
        long_w = random.choice([w for w in self.MEDIUM_WINDOWS if w > short_w] or [20])
        variant = random.randint(1, 4)

        if variant == 1:
            # Volatility ratio: short-term vol / long-term vol (vol expansion)
            short_vol = op('ts_std_dev', f(field), c(short_w))
            long_vol = op('ts_std_dev', f(field), c(long_w))
            return op('divide', short_vol, long_vol)
        elif variant == 2:
            # High-Low intraday range expansion vs rolling average range
            if 'high' in self.valid_fields and 'low' in self.valid_fields:
                bar_range = op('subtract', f('high'), f('low'))
                avg_range = op('ts_mean', bar_range.clone(), c(long_w))
                return op('divide', bar_range, avg_range)
            else:
                return op('divide', op('ts_std_dev', f(field), c(short_w)), c(1.0))
        elif variant == 3:
            # Directional volatility surge: delta * (short_vol / long_vol)
            delta = op('ts_delta', f(field), c(short_w))
            vol_ratio = op('divide', op('ts_std_dev', f(field), c(short_w)), op('ts_std_dev', f(field), c(long_w)))
            return op('multiply', delta, vol_ratio)
        else:
            # Volatility scale: percentile of current volatility over horizon
            vol = op('ts_std_dev', f(field), c(short_w))
            return op('ts_scale', vol, c(long_w))

    # --- 4. Volume-Price Flow & Divergence ---
    def _build_volume_price(self) -> ASTNode:
        price_field = random.choice(self.price_fields)
        window = random.choice(self.SHORT_WINDOWS + self.MEDIUM_WINDOWS)
        variant = random.randint(1, 5)

        if 'volume' not in self.valid_fields:
            return self._build_mean_reversion()

        if variant == 1:
            # Rolling correlation between price and volume
            return op('ts_corr', f(price_field), f('volume'), c(window))
        elif variant == 2:
            # Rolling covariance between price returns and volume
            delta_price = op('ts_delta', f(price_field), c(2))
            return op('ts_covariance', delta_price, f('volume'), c(window))
        elif variant == 3:
            # Price delta confirmed by volume z-score
            delta_price = op('ts_delta', f(price_field), c(min(5, window)))
            vol_z = op('ts_zscore', f('volume'), c(window))
            return op('multiply', delta_price, vol_z)
        elif variant == 4:
            # Flow divergence: Price drop with low volume (washout) or price up on high volume
            # reverse(zscore(price)) * zscore(volume)
            p_z = op('ts_zscore', f(price_field), c(window))
            v_z = op('ts_zscore', f('volume'), c(window))
            return op('multiply', op('reverse', p_z), v_z)
        else:
            # Product of normalized price change and volume ratio
            delta_price = op('ts_delta', f(price_field), c(2))
            avg_vol = op('ts_mean', f('volume'), c(window))
            vol_rel = op('divide', f('volume'), avg_vol)
            return op('multiply', delta_price, vol_rel)

    # --- 5. Dip & Rebound Archetype ---
    def _build_dip_rebound(self) -> ASTNode:
        field = 'close' if 'close' in self.valid_fields else random.choice(self.price_fields)
        window = random.choice(self.SHORT_WINDOWS + [20])
        variant = random.randint(1, 4)

        if variant == 1:
            # Deep oversold + volume spike: classic capitulation bottom
            oversold = op('reverse', op('ts_scale', f(field), c(window)))
            if 'volume' in self.valid_fields:
                vol_surge = op('ts_zscore', f('volume'), c(max(5, window // 2)))
                return op('multiply', oversold, vol_surge)
            return oversold
        elif variant == 2:
            # Multi-day oversold with 2-day positive tick (reversal trigger)
            oversold = op('reverse', op('ts_zscore', f(field), c(window)))
            rebound_tick = op('ts_delta', f(field), c(2))
            return op('multiply', op('rank', oversold), op('rank', rebound_tick))
        elif variant == 3:
            # Intraday hammer / wick dip recovery: (close - low) / (high - low)
            if 'high' in self.valid_fields and 'low' in self.valid_fields:
                numerator = op('subtract', f('close'), f('low'))
                denominator = op('add', op('subtract', f('high'), f('low')), c(0.0001))
                candle_pos = op('divide', numerator, denominator)
                # Combine with rolling oversold
                oversold = op('reverse', op('ts_zscore', f(field), c(window)))
                return op('multiply', candle_pos, oversold)
            else:
                return op('reverse', op('ts_zscore', f(field), c(window)))
        else:
            # Exhaustion dip: large negative delta damped by hump
            large_drop = op('reverse', op('ts_delta', f(field), c(window)))
            return op('hump', op('ts_zscore', large_drop, c(window)))

    # --- 6. Factor Combo Archetype ---
    def _build_factor_combo(self) -> ASTNode:
        # Pick two distinct scenarios and blend their signals
        sub_scenarios = ["mean_reversion", "momentum", "volume_price", "volatility_breakout"]
        picked = random.sample(sub_scenarios, 2)
        
        signal_a = self.generate(scenario=picked[0], apply_wrappers=False)
        signal_b = self.generate(scenario=picked[1], apply_wrappers=False)

        combo_op = random.choice(["add", "multiply"])
        if combo_op == "add":
            # Normalized addition: rank(A) + rank(B)
            return op('add', op('rank', signal_a), op('rank', signal_b))
        else:
            # Confluence: sign(A) * sign(B) or rank(A) * rank(B)
            return op('multiply', op('rank', signal_a), op('rank', signal_b))

    # --- Post-Processing / Wrappers ---
    def _apply_standard_wrappers(self, node: ASTNode) -> ASTNode:
        """
        Applies WorldQuant Brain standard alpha post-processing:
        1. Turnover reduction: ts_decay_linear or ts_mean or hump
        2. Final Outermost Neutralization: rank, zscore, normalize
           (CRITICAL: Neutralization must always be outermost so time-series smoothing does not destroy dollar neutrality)
        """
        # 1. Smoothing / Decay (75% chance)
        if random.random() < 0.75:
            smooth_op = random.choice(['ts_decay_linear', 'ts_mean', 'hump'])
            if smooth_op == 'hump':
                node = op('hump', node)
            else:
                decay_window = random.choice([3, 5, 8, 10, 15])
                node = op(smooth_op, node, c(decay_window))

        # 2. Final Outermost Neutralization (Always applied)
        neut_op = random.choice(['rank', 'zscore', 'normalize'])
        node = op(neut_op, node)

        return node
