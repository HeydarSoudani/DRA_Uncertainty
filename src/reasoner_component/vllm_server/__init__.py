"""Self-hosted vLLM servers: which models we serve and how, and running them.

``registry``  ``ServerSpec`` and ``SERVERS``: every model's server settings
``process``   ``VLLMServer``, ``VLLMServerManager``, ``serve``: start / stop / GPUs
``__main__``  ``python -m reasoner_component.vllm_server serve <model>`` (``list``)

The client that talks to a running server is ``reasoner_component.vllm.VLLMGenerator``.
Importing this package starts nothing and imports neither ``vllm`` nor ``torch``.
"""

from .process import VLLMServer, VLLMServerManager, serve
from .registry import SERVERS, ServerSpec, VLLMServerError, require_spec, server_spec

__all__ = [
    "SERVERS",
    "ServerSpec",
    "VLLMServerError",
    "server_spec",
    "require_spec",
    "VLLMServer",
    "VLLMServerManager",
    "serve",
]
