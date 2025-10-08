Qontinui Documentation
======================

Qontinui is a Python library for model-based GUI automation with intelligent state management and visual recognition.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   quickstart
   api/index

Overview
--------

Qontinui enables building robust GUI automation through:

* **Model-based state management** using MultiState
* **Visual recognition** with OpenCV template matching
* **JSON configuration** for defining automation workflows
* **Cross-platform support** (Windows, macOS, Linux)

Quick Example
-------------

.. code-block:: python

   from qontinui.json_executor import JSONRunner

   # Initialize runner
   runner = JSONRunner()

   # Load configuration
   runner.load_configuration("automation_config.json")

   # Execute automation
   success = runner.run(process_id="login_process", monitor_index=0)

Key Features
------------

* JSON-based automation configuration
* Template-based image matching
* Multi-state state management
* Process and state machine execution
* Cross-platform input control
* Hardware abstraction layer

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
