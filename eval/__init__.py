"""
eval — 轴承故障振动信号数据质量评估代码库

模块清单：
  dataset.py          数据集加载与管理
  models.py           1D-CNN 分类器
  eval_time_freq.py   时频域物理特征对比分析
  train_diagnosis.py  TRTR / TSTR 下游分类对比实验
  visualize_tsne.py   t-SNE 特征空间可视化
"""

from .dataset import BearingSignalDataset, make_dummy_datasets, build_dataloader
from .models import BearingCNN1D, build_cnn
from .eval_time_freq import run_time_freq_analysis
from .train_diagnosis import run_trtr_tstr
from .visualize_tsne import run_tsne_visualization

