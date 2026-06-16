SimpleCarrier
=============

:class:`~distributed_resource_optimization.SimpleCarrier` is the built-in lightweight
carrier. It runs all participants as asyncio tasks within a single process, with no
network or serialisation overhead. It is the recommended choice for prototyping, testing,
and single-machine simulations.

Core Types
----------

ActorContainer
~~~~~~~~~~~~~~

An :class:`~distributed_resource_optimization.ActorContainer` holds all
:class:`~distributed_resource_optimization.SimpleCarrier` instances and lets them find
each other by numeric ID. Create one container per simulation:

.. code-block:: python

   from distributed_resource_optimization import ActorContainer

   container = ActorContainer()

SimpleCarrier
~~~~~~~~~~~~~

Wraps an algorithm (or coordinator) and registers it with a container:

.. code-block:: python

   from distributed_resource_optimization import (
       SimpleCarrier, create_cohda_participant,
   )

   c1 = SimpleCarrier(container, create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]]))
   c2 = SimpleCarrier(container, create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]]))

Each carrier is automatically assigned a 1-indexed integer ID when it registers. Retrieve
it with :func:`~distributed_resource_optimization.cid`:

.. code-block:: python

   from distributed_resource_optimization import cid

   print(cid(c1))  # 1
   print(cid(c2))  # 2

Sending Messages
----------------

Use :meth:`~distributed_resource_optimization.SimpleCarrier.send_to_other` to dispatch a
message to another carrier in the same container. The message is delivered asynchronously
as an asyncio Task:

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import create_cohda_start_message

   async def main():
       start = create_cohda_start_message([1.2, 2.0, 3.0])
       c1.send_to_other(start, cid(c2))
       await container.done_event.wait()   # block until all tasks finish

   asyncio.run(main())

Express API
-----------

For quick experiments, skip creating the container and carriers yourself.
:func:`~distributed_resource_optimization.start_distributed_optimization` wraps everything
in a single call:

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import (
       create_cohda_participant,
       create_cohda_start_message,
       start_distributed_optimization,
   )

   async def main():
       actor1 = create_cohda_participant(1, [[0.0, 1, 2], [1, 2, 3]])
       actor2 = create_cohda_participant(2, [[0.0, 1, 2], [1, 2, 3]])
       start = create_cohda_start_message([1.2, 2.0, 3.0])
       await start_distributed_optimization([actor1, actor2], start)

   asyncio.run(main())

For coordinated algorithms (e.g. ADMM), use
:func:`~distributed_resource_optimization.start_coordinated_optimization`:

.. code-block:: python

   import asyncio
   from distributed_resource_optimization import (
       create_admm_flex_actor_one_to_many,
       create_sharing_target_distance_admm_coordinator,
       create_admm_sharing_data, create_admm_start,
       start_coordinated_optimization,
   )

   async def main():
       flex1 = create_admm_flex_actor_one_to_many(10.0, [0.1,  0.5, -1.0])
       flex2 = create_admm_flex_actor_one_to_many(15.0, [0.1,  0.5, -1.0])
       flex3 = create_admm_flex_actor_one_to_many(10.0, [-1.0, 0.0,  1.0])
       coordinator = create_sharing_target_distance_admm_coordinator()
       start = create_admm_start(create_admm_sharing_data([-4.0, 0.0, 6.0], [5, 1, 1]))
       await start_coordinated_optimization([flex1, flex2, flex3], coordinator, start)

   asyncio.run(main())

Awaitable Messages
------------------

When a participant needs a response before continuing (e.g. the ADMM x-update),
:meth:`~distributed_resource_optimization.SimpleCarrier.send_awaitable` returns an
``asyncio.Future`` that resolves to the reply:

.. code-block:: python

   future = c1.send_awaitable(my_request, cid(c2))
   response = await future

The ADMM coordinator uses this internally to collect all x-updates in parallel.

.. note::

   All message dispatches are asyncio Tasks.  The :attr:`done_event` on the container is
   set when the active-task counter reaches zero, signalling that the distributed run has
   finished.

See Also
--------

- :class:`~distributed_resource_optimization.SimpleCarrier`,
  :class:`~distributed_resource_optimization.ActorContainer`,
  :func:`~distributed_resource_optimization.cid`
- :func:`~distributed_resource_optimization.start_distributed_optimization`,
  :func:`~distributed_resource_optimization.start_coordinated_optimization`
- :doc:`mango` — mango-agents carrier for TCP networking
