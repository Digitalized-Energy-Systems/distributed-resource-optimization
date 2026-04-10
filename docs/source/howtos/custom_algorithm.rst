How To: Implement a Custom Algorithm
======================================

The algorithm abstraction has a single required method.  Implementing it lets your
algorithm participate in any carrier — ``SimpleCarrier``, ``MangoCarrier``, or one you
write yourself.

The Interface
-------------

All distributed algorithms extend
:class:`~distributed_resource_optimization.DistributedAlgorithm` and implement
:meth:`~distributed_resource_optimization.DistributedAlgorithm.on_exchange_message`:

.. code-block:: python

   from distributed_resource_optimization import DistributedAlgorithm

   class MyAlgorithm(DistributedAlgorithm):
       async def on_exchange_message(self, carrier, message_data, meta):
           # Your logic here
           ...

.. list-table::
   :header-rows: 1
   :widths: 15 20 50

   * - Argument
     - Type
     - Description
   * - ``carrier``
     - :class:`~distributed_resource_optimization.Carrier`
     - Use this to send replies or inspect neighbours
   * - ``message_data``
     - Any
     - The deserialized message payload
   * - ``meta``
     - ``dict``
     - Transport metadata — e.g. ``{"sender": 1, "message_id": uuid}``

Step-by-Step Example — Echo Algorithm
---------------------------------------

As a minimal example: an algorithm that counts messages received and echoes each one back.

**1. Define the algorithm class**

.. code-block:: python

   from distributed_resource_optimization import (
       DistributedAlgorithm,
       OptimizationMessage,
   )

   class EchoMessage(OptimizationMessage):
       def __init__(self, payload: str):
           self.payload = payload

   class EchoAlgorithm(DistributedAlgorithm):
       def __init__(self):
           self.count = 0

       async def on_exchange_message(self, carrier, message_data, meta):
           self.count += 1
           print(f"Received #{self.count}: {message_data.payload}")
           carrier.reply_to_other(EchoMessage(f"ACK: {message_data.payload}"), meta)

**2. Run it with SimpleCarrier**

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import (
       ActorContainer, SimpleCarrier, cid,
   )

   async def main():
       container = ActorContainer()
       c1 = SimpleCarrier(container, EchoAlgorithm())
       c2 = SimpleCarrier(container, EchoAlgorithm())

       c1.send_to_other(EchoMessage("hello"), cid(c2))
       await container.done_event.wait()

       print(f"c2 received {c2.actor.count} message(s)")

   asyncio.run(main())

Coordinated Algorithms
-----------------------

For algorithms that require a central coordinator (like ADMM), implement a
:class:`~distributed_resource_optimization.Coordinator` alongside your algorithm:

.. code-block:: python

   from distributed_resource_optimization import Coordinator

   class MyCoordinator(Coordinator):
       async def start_optimization(self, carrier, message_data, meta):
           participants = carrier.others("coordinator")
           # ... send messages, collect replies, return results
           return []

The coordinator is automatically given its own ``SimpleCarrier`` (or mango agent) and
becomes the entry point for
:func:`~distributed_resource_optimization.start_coordinated_optimization`.

Practical Tips
--------------

**Keep algorithm state in the class.**
Since each ``SimpleCarrier`` dispatches messages via asyncio tasks, mutable attributes of
your algorithm class are effectively actor state — concurrent access within a single task
is naturally safe.

**Use** ``send_awaitable`` **for request/reply patterns.**
If your algorithm needs a response before proceeding:

.. code-block:: python

   future = carrier.send_awaitable(MyRequest(data), target_id)
   response = await future

The ADMM coordinator uses this to collect all x-updates in parallel via
``asyncio.gather(*futures)``.

**Termination.**
There is no built-in termination protocol.  Implement convergence detection inside
``on_exchange_message`` and simply stop sending messages when done. The ``done_event``
on the ``ActorContainer`` fires automatically once all in-flight tasks have finished.

See Also
--------

- :class:`~distributed_resource_optimization.DistributedAlgorithm`,
  :class:`~distributed_resource_optimization.Coordinator`
- :class:`~distributed_resource_optimization.Carrier`,
  :class:`~distributed_resource_optimization.OptimizationMessage`
- :doc:`custom_carrier`
