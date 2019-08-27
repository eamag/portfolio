from api.core_api import MyFlaskApp
from loguru import logger
from datetime import datetime


@logger.catch
def __run_app():
    logger.info(f'start app at {datetime.now()}')
    app_inited = MyFlaskApp()
    app_inited.app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    __run_app()
