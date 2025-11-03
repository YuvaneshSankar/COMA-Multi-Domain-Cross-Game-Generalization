

import numpy as np
import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class AnalysisEngine:
    """Core analysis functionality"""

    def __init__(self, results_dir: str = './results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def ablation_study_results(self, ablation_configs: Dict[str, Dict]) -> Dict[str, float]:
        """Compute ablation study impact"""
        return {name: config.get('performance', 0.0) for name, config in ablation_configs.items()}

    def generate_report(self, results: Dict[str, Any], title: str = "Training Report") -> str:
        """Generate markdown report"""
        report = f"# {title}\n\n"
        report += f"Generated: {Path.cwd()}\n\n"

        for section, data in results.items():
            report += f"## {section}\n\n"
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, float):
                        report += f"- {key}: {val:.4f}\n"
                    else:
                        report += f"- {key}: {val}\n"
            report += "\n"

        return report

    def save_analysis(self, analysis: Dict, filename: str):
        """Save analysis to JSON"""
        path = self.results_dir / filename
        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {path}")


class StatisticalTests:
    """Statistical analysis methods"""

    @staticmethod
    def significance_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Simple t-test comparison"""
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(group1, group2)
        return {'t_statistic': t_stat, 'p_value': p_value}

    @staticmethod
    def confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
        """Compute confidence interval"""
        import scipy.stats as stats
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return (mean - ci, mean + ci)


if __name__ == "__main__":
    print("Analysis module ready")
