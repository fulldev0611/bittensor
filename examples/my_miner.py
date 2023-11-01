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

import torch
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel


class MyMiner(bittensor.HuggingFaceMiner):
    arg_prefix: str = "my_miner"
    assistant_label: str = ""
    user_label: str = ""
    system_label: str = ""

    
    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self.process_history(messages)
        prompt = history[-1][-1]
        if len(history) == 1:
            history = []
        
        print("Prompt:   " +  prompt)
        generation = "I am a Danila Bot"

        bittensor.logging.debug(
            "Message: " + str(messages).replace("<", "-").replace(">", "-")
        )
        bittensor.logging.debug(
            "Generation: " + str(generation).replace("<", "-").replace(">", "-")
        )
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    MyMiner().run()
