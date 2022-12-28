from pathlib import Path
from utils import read_json


class ConfigParser:
    def __init__(self, config, run_id=None):

        self.config = read_json(config)

        save_dir = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']
        self._save_dir = save_dir / 'models' / exper_name / run_id

        # make directory for saving checkpoints and log.
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self.module_name = self.config['arch']['type']
        self.module_args = self.config['arch']['args']
        self.data_loader = self.config['data_loader']
        self.optimizer = self.config['optimizer']
        self.loss_fn = self.config['loss']
        self.metric_fn = self.config['metrics']
        self.lr_scheduler = self.config['lr_scheduler']
        self.trainer = self.config['trainer']

    @property
    def save_dir(self):
        return self._save_dir

