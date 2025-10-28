API Reference
=============

This section contains the API documentation for Qontinui.

.. toctree::
   :maxdepth: 2

   json_executor
   model
   wrappers
   config

JSON Executor
-------------

.. automodule:: qontinui.json_executor.json_runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qontinui.json_executor.config_parser
   :members:
   :undoc-members:
   :show-inheritance:

Action Executors
----------------

The action execution system has been refactored into a modular architecture.
For the legacy monolithic implementation, see :mod:`qontinui.json_executor.action_executor` (deprecated).

.. automodule:: qontinui.action_executors
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qontinui.action_executors.delegating_executor
   :members:
   :undoc-members:
   :show-inheritance:
