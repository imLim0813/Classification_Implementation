import wandb
from typing import Dict


class Logger:
    """
    wandb 라이브러리를 이용하여 학습 로그를 기록하기 위한 클래스

    :param config_file: 실험 조건을 담고 있는 dict 객체
    :param p_name: 프로젝트 명 문자열
    :param r_name: 실험 명 문자열
    """
    def __init__(self, config_file: Dict, p_name: str, r_name: str):
        self.config = config_file
        self.logger = wandb.init(dir='./', project=p_name, name=r_name, config=self.config)

    def record(self, log_file: Dict):
        """
        로그를 딕셔너리 형태로 받아 학습을 기록하는 함수

        :param log_file: 기록할 로그들을 담은 딕셔너리 객체
        """
        self.logger.log(log_file)

    def finish(self):
        """
        학습 종료시 로그를 기록하는 wandb를 종료시키기 위한 함수
        """
        self.logger.finish()