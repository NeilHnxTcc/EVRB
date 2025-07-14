"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from omegaconf import OmegaConf
from transformers import CLIPImageProcessor

class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)


class ClipImageEvalProcessor(BaseProcessor):
    def __init__(self, proc_type, do_normalize=True):
        super().__init__()
        self.transform = CLIPImageProcessor.from_pretrained(proc_type)
        self.transform.do_normalize = True if do_normalize else False

    def __call__(self, item):
        return self.transform.preprocess(item, return_tensors='pt')['pixel_values'][0]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        proc_type = cfg.get("proc_type", r'openai/clip-vit-large-patch14')

        do_normalize = cfg.get("do_normalize", True)

        return cls(proc_type=proc_type, do_normalize=do_normalize)

