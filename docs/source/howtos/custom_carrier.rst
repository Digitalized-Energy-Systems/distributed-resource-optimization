How To: Implement a Custom Carrier
====================================

A *carrier* handles the transport of messages between algorithm participants. Implementing
a custom carrier lets you plug in any communication backend — an actor framework, an
event bus, a message queue, etc. — without changing any algorithm code.

The Interface
-------------

All carriers extend :class:`~distributed_resource_optimization.carrier.core.Carrier` and implement
four abstract methods:

.. code-block:: python

   from distributed_resource_optimization import Carrier

   class MyCarrier(Carrier):

       def send_to_other(self, content, receiver):
           """Dispatch content to receiver (fire-and-forget). Return a Task."""
           ...

       def reply_to_other(self, content, meta):
           """Reply to the sender identified in meta. Return a Task."""
           ...

       def send_awaitable(self, content, receiver):
           """Send content and return a Future that resolves to the reply."""
           ...

       def others(self):
           """Return the addresses of every other participant this one talks to."""
           ...

Minimal Example — In-Memory Bus
---------------------------------

Here is a minimal custom carrier that only records messages in a dictionary — useful for
unit-testing an algorithm's outgoing messages. It never delivers anything, so its
``send_awaitable`` future resolves to ``None`` instead of a real reply:

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import Carrier

   class BusCarrier(Carrier):
       """Simple test carrier backed by an in-memory message log."""

       def __init__(self, aid: int, bus: dict):
           self.aid = aid
           self.bus = bus  # shared dict: receiver_id -> list[message]

       def send_to_other(self, content, receiver):
           self.bus.setdefault(receiver, []).append(content)
           future = asyncio.get_event_loop().create_future()
           future.set_result(None)
           return future

       def reply_to_other(self, content, meta):
           return self.send_to_other(content, meta["sender"])

       def send_awaitable(self, content, receiver):
           self.send_to_other(content, receiver)
           future = asyncio.get_event_loop().create_future()
           future.set_result(None)  # no real transport, so no reply
           return future

       def others(self):
           return [k for k in self.bus if k != self.aid]

Tips
----

**Route replies by message ID.**
The request/reply pattern (used by ADMM) relies on the carrier preserving the
``message_id`` from the outgoing meta and routing the reply back to the waiting future.
See :class:`~distributed_resource_optimization.carrier.simple.SimpleCarrier` for a reference
implementation.

**Thread safety.**
If your carrier dispatches across real threads (not asyncio tasks), protect shared state
with locks or use thread-safe queues.

**Termination.**
If you need a done-event equivalent, track in-flight message counts and signal completion
when the count drops to zero — exactly as ``ActorContainer`` does.

See Also
--------

- :class:`~distributed_resource_optimization.carrier.core.Carrier` — abstract base class
- :class:`~distributed_resource_optimization.carrier.simple.SimpleCarrier` — asyncio reference implementation
- :class:`~distributed_resource_optimization.carrier.mango.MangoCarrier` — mango-agents implementation
- :doc:`custom_algorithm`
