import unittest, tempfile, os
from . import TransformerHeadConfig, SubwordClassificationHead, CLSRegressionHead
class TestTransformerHeadConfig(unittest.TestCase):

    def test_config(self):
        config = TransformerHeadConfig({'head_name': "subword_classification_head", 'head_task': "Random NER Dataset"})

        self.assertEqual(config.head_name,"subword_classification_head")
        self.assertEqual(config.head_task,"Random NER Dataset")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.to_json_file(os.path.join(temp_dir, f"{config}.json"))
            loaded_config = TransformerHeadConfig.from_json_file(os.path.join(temp_dir, f"{config}.json"))
            #print(os.path.join(temp_dir, f"{config}.json"))
            self.assertEqual(loaded_config.to_json_string(), config.to_json_string())
        #
        # self.assertEqual(config.to_json_string(), "")

class TestSubwordClassificationHead(unittest.TestCase):
    def test_loading_head(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            head = SubwordClassificationHead('test_task',
                                             labels=['O', 'B-Drug'],
                                             hidden_size=768,
                                             hidden_dropout_prob=.1)
            head.save(str(temp_dir))
            head2 = SubwordClassificationHead('test_task', labels=head.config.labels)
            head2.from_pretrained(temp_dir)
            self.assertEqual(dict(head.config), dict(head2.config))

class TestCLSRegressionHead(unittest.TestCase):

    def test_loading_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            head = CLSRegressionHead('test_task',
                                     hidden_size=768,
                                     hidden_dropout_prob=.1)
            head.save(str(temp_dir))
            head2 = CLSRegressionHead('test_task')
            head2.from_pretrained(temp_dir)
            self.assertEqual(dict(head.config), dict(head2.config))