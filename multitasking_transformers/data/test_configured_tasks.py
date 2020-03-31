import unittest

from . import NERDataset
from transformers import BertTokenizer, AlbertTokenizer


class TestConfiguredTasks(unittest.TestCase):

    def test_configured_tasks(self):
        from multi_tasking_transformers.data.configured_tasks import get_all_configured_tasks

        TASKS = get_all_configured_tasks("/home/aymulyar/development/multitasking_transformers/experiments/data")

        self.assertIsInstance(TASKS, dict)
