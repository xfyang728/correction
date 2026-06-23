"""
数据模型 — SQLAlchemy ORM 定义。
"""

import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Task(Base):
    """一次批改任务"""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, index=True)
    class_name = Column(String(50), nullable=True)
    date_str = Column(String(20), nullable=True)
    seq = Column(String(10), nullable=True)
    total_chars = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    uncertain_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)
    result_json = Column(JSON, nullable=True)       # 完整批改结果
    annotated_pdf = Column(String(500), nullable=True)  # 批注后 PDF 路径
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # 可观测性字段
    needs_review = Column(Boolean, default=False)       # 低置信度占比>30% 自动标记
    review_done = Column(Boolean, default=False)        # 教师已完成复核
    avg_confidence = Column(Float, nullable=True)       # OCR 平均置信度
    low_conf_ratio = Column(Float, nullable=True)       # 低置信度字占比
    process_time = Column(Float, nullable=True)         # 处理耗时（秒）


class Answer(Base):
    """标准答案记录"""
    __tablename__ = "answers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    class_name = Column(String(50), nullable=True)
    date_str = Column(String(20), nullable=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ReviewCorrection(Base):
    """人工复核修正记录 — 积累为微调训练集"""
    __tablename__ = "review_corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, nullable=False, index=True)      # 关联 Task
    char_index = Column(Integer, nullable=False)               # 在 result_json 中的索引
    original_char = Column(String(20), nullable=True)          # OCR 识别的字符
    original_status = Column(String(20), nullable=True)        # 原始状态
    corrected_char = Column(String(20), nullable=True)         # 教师修正的字符
    corrected_status = Column(String(20), nullable=False)      # 教师修正的状态
    bbox_pixel = Column(JSON, nullable=True)                   # 字符坐标（训练数据用）
    confidence = Column(Float, nullable=True)                  # 原始 OCR 置信度
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db(db_path: str = "data/marker.db"):
    """初始化数据库，返回 Session 类。"""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
