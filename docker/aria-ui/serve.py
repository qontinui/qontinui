"""Wrapper to patch Aria-UI tokenizer before vllm starts.

The Aria-UI tokenizer_config.json includes image_token="<|img|>" but
LlamaTokenizerFast doesn't expose it as an attribute. The transformers
AriaProcessor.__init__ expects tokenizer.image_token to exist.

This script patches the tokenizer class to add the missing attributes,
then launches vllm's OpenAI-compatible server.
"""
import transformers.models.llama.tokenization_llama_fast as llama_fast

# Patch LlamaTokenizerFast to surface image_token from tokenizer_config
_orig_init = llama_fast.LlamaTokenizerFast.__init__

def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    # Surface image_token and image_token_id from tokenizer_config.json
    init_kwargs = getattr(self, 'init_kwargs', {})
    if not hasattr(self, 'image_token'):
        token = init_kwargs.get('image_token') or kwargs.get('image_token')
        if token:
            self.image_token = token
    if not hasattr(self, 'image_token_id'):
        token = getattr(self, 'image_token', None)
        if token:
            ids = self.encode(token, add_special_tokens=False)
            if ids:
                self.image_token_id = ids[0]

llama_fast.LlamaTokenizerFast.__init__ = _patched_init

# Guard against spawn-based multiprocessing re-executing this script
if __name__ == "__main__":
    import sys

    from vllm.entrypoints.openai.api_server import FlexibleArgumentParser, run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(sys.argv[1:])

    import uvloop
    uvloop.run(run_server(args))
