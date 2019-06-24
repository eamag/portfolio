from sqlalchemy import Column, String, Boolean, Integer, Time, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, String, MetaData


def connect_default():
    db_string = "postgres://login:password@localhost:5432/dbname"

    db = create_engine(db_string)
    meta = MetaData(db)
    Session = sessionmaker(db)
    session = Session()
    base = declarative_base()
    return meta, session, base, db


meta, session, base, db = connect_default()


class Event(base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    link = Column(String)
    place = Column(String)
    time = Column(DateTime)
    link_reg = Column(String)
    chose_1 = Column(Boolean, default=False)
    chose_2 = Column(Boolean, default=False)
    th_business = Column(Boolean, default=False)
    th_career = Column(Boolean, default=False)
    th_tech = Column(Boolean, default=False)
    th_arts = Column(Boolean, default=False)
    th_chill = Column(Boolean, default=False)
    is_confirmed = Column(Boolean, default=False)


class User(base):
    __tablename__ = 'users'

    uid = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    is_sub = Column(Boolean, default=False)
    chose_1 = Column(Boolean, default=False)
    chose_2 = Column(Boolean, default=False)
    th_business = Column(Boolean, default=False)
    th_career = Column(Boolean, default=False)
    th_tech = Column(Boolean, default=False)
    th_arts = Column(Boolean, default=False)
    th_chill = Column(Boolean, default=False)


base.metadata.create_all(db)
