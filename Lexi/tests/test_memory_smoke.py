from lexi.memory.memory_core import MemoryManager
from lexi.memory.memory_store_json import MemoryStoreJSON
from lexi.memory.memory_types import MemoryShard


def test_memory_smoke(tmp_path):
    store_path = tmp_path / "lexi_memory.jsonl"
    store = MemoryStoreJSON(store_path)
    memory = MemoryManager(store)

    shard_one = MemoryShard(role="user", content="Hello Lexi!")
    shard_two = MemoryShard(role="assistant", content="Hi there!")

    memory.remember(shard_one)
    memory.remember(shard_two)

    assert memory.recent(2)
    assert len(memory.all()) >= 2

    # Ensure delete does not raise even if the id is not yet persisted.
    memory.delete(shard_one.created_at)
