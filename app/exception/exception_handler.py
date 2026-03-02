from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.exception.exception import http_exception_handler, integrity_error_handler, sqlalchemy_error_handler, \
    general_exception_handler, CustomException, custom_exception_handler


def register_exception_handler(app: FastAPI):
    """
    注册全局异常处理器
    :param app:
    :return:
    """
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(IntegrityError, integrity_error_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_error_handler)
    app.add_exception_handler(CustomException, custom_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)