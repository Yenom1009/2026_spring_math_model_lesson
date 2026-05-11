"""
实验结果记录模块
将实验结果记录到表格文件中，方便后续在报告中比较和分析
"""
import os
import csv
import json
from datetime import datetime
import numpy as np


class ExperimentLogger:
    """实验结果记录器"""
    
    def __init__(self, log_dir='experiment_logs'):
        """
        初始化记录器
        
        参数:
            log_dir: 日志文件保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建主日志文件
        self.summary_file = os.path.join(log_dir, 'experiment_summary.csv')
        self.detailed_file = os.path.join(log_dir, 'experiment_detailed.json')
        
        # 初始化CSV文件（如果不存在）
        if not os.path.exists(self.summary_file):
            self._init_summary_csv()
        
        # 存储当前会话的所有实验结果
        self.session_results = []
        self.session_start_time = datetime.now()
    
    def _init_summary_csv(self):
        """初始化汇总CSV文件"""
        headers = [
            'Timestamp', 'Experiment', 'Curve', 'Method', 
            'RMSE_Samples', 'RMSE_True', 'Max_Deviation', 
            'Mean_Distance', 'Hausdorff', 'Smoothness_Energy',
            'Curvature_Std', 'Curvature_Max', 'Notes'
        ]
        with open(self.summary_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_experiment(self, experiment_name, curve_name, method_name, 
                      metrics, notes=''):
        """
        记录单个实验结果
        
        参数:
            experiment_name: 实验名称（如 'Exp1_Parameterization'）
            curve_name: 曲线名称（如 'Heart'）
            method_name: 方法名称（如 'Chord'）
            metrics: 指标字典
            notes: 备注信息
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 提取关键指标（如果不存在则用N/A）
        row = [
            timestamp,
            experiment_name,
            curve_name,
            method_name,
            metrics.get('rmse_samples', 'N/A'),
            metrics.get('rmse_true', 'N/A'),
            metrics.get('max_dev_samples', 'N/A'),
            metrics.get('mean_dist_samples', 'N/A'),
            metrics.get('hausdorff', 'N/A'),
            metrics.get('smoothness_energy', 'N/A'),
            metrics.get('curvature_std', 'N/A'),
            metrics.get('curvature_max', 'N/A'),
            notes
        ]
        
        # 写入CSV
        with open(self.summary_file, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # 存储到会话结果
        result = {
            'timestamp': timestamp,
            'experiment': experiment_name,
            'curve': curve_name,
            'method': method_name,
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                       for k, v in metrics.items()},
            'notes': notes
        }
        self.session_results.append(result)
    
    def log_comparison(self, experiment_name, curve_name, methods_metrics, notes=''):
        """
        记录对比实验结果（多个方法）
        
        参数:
            experiment_name: 实验名称
            curve_name: 曲线名称
            methods_metrics: 字典 {method_name: metrics_dict, ...}
            notes: 备注信息
        """
        for method_name, metrics in methods_metrics.items():
            self.log_experiment(experiment_name, curve_name, method_name, 
                              metrics, notes)
    
    def create_comparison_table(self, experiment_name, output_file=None):
        """
        为特定实验创建对比表格（Markdown格式）
        
        参数:
            experiment_name: 实验名称
            output_file: 输出文件路径（默认在log_dir下）
        """
        if output_file is None:
            output_file = os.path.join(self.log_dir, 
                                      f'{experiment_name}_comparison.md')
        
        # 读取CSV文件
        rows = []
        with open(self.summary_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Experiment'] == experiment_name:
                    rows.append(row)
        
        if not rows:
            print(f"警告：没有找到实验 {experiment_name} 的数据")
            return
        
        # 按曲线分组
        curves = {}
        for row in rows:
            curve = row['Curve']
            if curve not in curves:
                curves[curve] = []
            curves[curve].append(row)
        
        # 生成Markdown表格
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {experiment_name} - 实验结果对比\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for curve_name, curve_rows in curves.items():
                f.write(f"## {curve_name}\n\n")
                
                # 表头
                f.write("| 方法 | RMSE(样本) | RMSE(真实) | 最大偏差 | 平均距离 | Hausdorff | 平滑能量 |\n")
                f.write("|------|-----------|-----------|---------|---------|-----------|----------|\n")
                
                # 数据行
                for row in curve_rows:
                    method = row['Method']
                    rmse_s = self._format_number(row['RMSE_Samples'])
                    rmse_t = self._format_number(row['RMSE_True'])
                    max_dev = self._format_number(row['Max_Deviation'])
                    mean_d = self._format_number(row['Mean_Distance'])
                    hausd = self._format_number(row['Hausdorff'])
                    smooth = self._format_number(row['Smoothness_Energy'])
                    
                    f.write(f"| {method} | {rmse_s} | {rmse_t} | {max_dev} | "
                           f"{mean_d} | {hausd} | {smooth} |\n")
                
                f.write("\n")
        
        print(f"对比表格已保存到: {output_file}")
        return output_file
    
    def _format_number(self, value):
        """格式化数字显示"""
        if value == 'N/A' or value == '':
            return 'N/A'
        try:
            num = float(value)
            if num < 0.01:
                return f"{num:.6f}"
            elif num < 1:
                return f"{num:.4f}"
            elif num < 100:
                return f"{num:.2f}"
            else:
                return f"{num:.1f}"
        except:
            return str(value)
    
    def save_session_summary(self):
        """保存当前会话的详细结果（JSON格式）"""
        session_data = {
            'session_start': self.session_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_end': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_experiments': len(self.session_results),
            'results': self.session_results
        }
        
        # 读取现有数据
        if os.path.exists(self.detailed_file):
            with open(self.detailed_file, 'r', encoding='utf-8') as f:
                try:
                    all_sessions = json.load(f)
                except:
                    all_sessions = []
        else:
            all_sessions = []
        
        # 添加新会话
        all_sessions.append(session_data)
        
        # 保存
        with open(self.detailed_file, 'w', encoding='utf-8') as f:
            json.dump(all_sessions, f, indent=2, ensure_ascii=False)
        
        print(f"\n会话结果已保存到: {self.detailed_file}")
    
    def generate_all_comparison_tables(self):
        """为所有实验生成对比表格"""
        # 读取所有实验名称
        experiments = set()
        with open(self.summary_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                experiments.add(row['Experiment'])
        
        # 为每个实验生成表格
        for exp in experiments:
            self.create_comparison_table(exp)
    
    def print_summary(self):
        """打印当前会话的汇总信息"""
        print("\n" + "="*70)
        print("实验结果汇总")
        print("="*70)
        print(f"会话开始时间: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总实验数: {len(self.session_results)}")
        print(f"\n结果文件:")
        print(f"  - CSV汇总: {self.summary_file}")
        print(f"  - JSON详细: {self.detailed_file}")
        print(f"  - Markdown对比表: {self.log_dir}/*_comparison.md")
        print("="*70 + "\n")


def create_latex_table(csv_file, experiment_name, output_file=None):
    """
    从CSV文件创建LaTeX表格（用于论文报告）
    
    参数:
        csv_file: CSV文件路径
        experiment_name: 实验名称
        output_file: 输出文件路径
    """
    if output_file is None:
        output_file = csv_file.replace('.csv', f'_{experiment_name}_latex.tex')
    
    rows = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Experiment'] == experiment_name:
                rows.append(row)
    
    if not rows:
        print(f"警告：没有找到实验 {experiment_name} 的数据")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{experiment_name} 实验结果对比}}\n")
        f.write("\\begin{tabular}{llcccccc}\n")
        f.write("\\hline\n")
        f.write("曲线 & 方法 & RMSE(样本) & RMSE(真实) & 最大偏差 & "
               "平均距离 & Hausdorff & 平滑能量 \\\\\n")
        f.write("\\hline\n")
        
        for row in rows:
            curve = row['Curve']
            method = row['Method']
            rmse_s = row['RMSE_Samples']
            rmse_t = row['RMSE_True']
            max_dev = row['Max_Deviation']
            mean_d = row['Mean_Distance']
            hausd = row['Hausdorff']
            smooth = row['Smoothness_Energy']
            
            f.write(f"{curve} & {method} & {rmse_s} & {rmse_t} & {max_dev} & "
                   f"{mean_d} & {hausd} & {smooth} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX表格已保存到: {output_file}")


if __name__ == '__main__':
    # 测试记录器
    logger = ExperimentLogger()
    
    # 模拟记录一些实验结果
    print("测试实验记录器...")
    
    # 实验1：参数化对比
    metrics_uniform = {
        'rmse_samples': 0.123,
        'rmse_true': 0.456,
        'max_dev_samples': 0.234,
        'mean_dist_samples': 0.111,
        'hausdorff': 0.345,
        'smoothness_energy': 12.34,
        'curvature_std': 0.056,
        'curvature_max': 0.789
    }
    
    metrics_chord = {
        'rmse_samples': 0.089,
        'rmse_true': 0.234,
        'max_dev_samples': 0.156,
        'mean_dist_samples': 0.078,
        'hausdorff': 0.234,
        'smoothness_energy': 8.92,
        'curvature_std': 0.045,
        'curvature_max': 0.567
    }
    
    logger.log_experiment('Exp1_Parameterization', 'Heart', 'Uniform', 
                         metrics_uniform, '均匀参数化')
    logger.log_experiment('Exp1_Parameterization', 'Heart', 'Chord', 
                         metrics_chord, '弦长参数化')
    
    # 生成对比表格
    logger.create_comparison_table('Exp1_Parameterization')
    
    # 保存会话
    logger.save_session_summary()
    
    # 打印汇总
    logger.print_summary()
    
    print("\n测试完成！请查看 experiment_logs/ 目录")
