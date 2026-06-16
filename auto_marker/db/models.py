"""
数据模型 — SQLAlchemy ORM 定义。
"""

import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
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


class Answer(Base):
    """标准答案记录"""
    __tablename__ = "answers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    class_name = Column(String(50), nullable=True)
    date_str = Column(String(20), nullable=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db(db_path: str = "data/marker.db"):
    """初始化数据库，返回 Session 类。"""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
