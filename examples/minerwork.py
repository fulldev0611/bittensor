# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import bittensor
from typing import List, Dict
from torch import FloatTensor


class MinerWorker(bittensor.BasePromptingMiner):
    @classmethod
    def check_config(cls, config: "bittensor.Config"):
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--gpt4all.model",
            type=str,
            help="Path to pretrained gpt4all model in ggml format.",
        )
            

    def __init__(self):
        super(MinerWorker, self).__init__()
        print(self.config)
        

    def backward(
        self, messages: List[Dict[str, str]], response: str, rewards: FloatTensor
    ) -> str:
        pass

    @staticmethod
    def _process_history(history: List[dict]) -> str:
        processed_history = ""
        for message in history:
            if message["role"] == "system":
                processed_history += "system: " + message["content"] + "\n"
            if message["role"] == "assistant":
                processed_history += "assistant: " + message["content"] + "\n"
            if message["role"] == "user":
                processed_history += "user: " + message["content"] + "\n"
        return processed_history

    def forward(self, messages: List[Dict[str, str]]) -> str:
        bittensor.logging.info("messages", str(messages))
        history = self._process_history(messages)
        bittensor.logging.info("history", str(history))
        # resp = self.model(history)
        resp = "It is generation from Danila"
        bittensor.logging.info("response", str(resp))
        return resp


if __name__ == "__main__":
    bittensor.utils.version_checking()
    MinerWorker().run()
